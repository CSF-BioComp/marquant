import cv2 as cv
import imutils
import numpy as np
import pandas as pd
import os
import logging
from configfy import configfy as cfy

import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from functools import partial
from tqdm import tqdm

from pudb import set_trace as st


def slide_experiment(image_folder_path, single_process):

    # Find all tif images in directory
    imgfiles = glob.glob(os.path.join(image_folder_path, '**/*.tif'), recursive=True) 
    N = len(imgfiles)

    if N == 0:
        logging.warn('No images found!')
    else:
        # Run on multiple processes and track using tqdm
        pack = zip(imgfiles, range(N))
        tqdm_kwargs = {'desc': 'Image Quantification ', 'total': N} 
        partial_results = multiprocess_tracking(slide_quantifiaction_wrapper, pack, tqdm_kwargs=tqdm_kwargs, single_process=single_process)
        
        #  Join partial results and order table
        logging.debug('Joining partial results ...') 
        exp_results = pd.concat(partial_results, sort=False)
        sort_order = ['plate_id', 'cell', 'row', 'column']
        exp_results.sort_values(sort_order, inplace=True)
        
        # Write results to csv
        output_path = os.path.join(image_folder_path, 'results.csv')
        logging.info('Writing results to %s ...', output_path)
        exp_results.to_csv(output_path, index=False)


def slide_quantifiaction_wrapper(pack):
    try:
        return slide_quantifiaction(pack[0], pack[1])
    except Exception as e:
        logging.error(f'Quantifying image {pack[0]} failed!')
        return pd.DataFrame()


@cfy
def slide_quantifiaction(image_path, img_id, ref_image_path='reference_plate.png', grid_positions=[]):
    """Quantify the micro array cells in given image using a reference image
    
    Arguments:
        image_path {string} -- Path to image file that should be quantified
        img_id {int} -- Unique identifier for this image
    
    Keyword Arguments:
        ref_image_path {str} -- Reference image that is used to align given target image (default: {'reference_plate.png'})
    """
    # Import for each process
    import matplotlib
    from matplotlib import pyplot as plt

    # Prepare the output dataframe
    quantified_plate = pd.DataFrame()

    # Load target and reference image
    logging.info(f'Loading image {image_path} ...')
    ref_img = cv.imread(ref_image_path)
    target_img = cv.imread(image_path)

    # Extract bounding box of reference image
    gray_ref_img = cv.cvtColor(ref_img,cv.COLOR_BGR2GRAY) 
    _, ref_mask = cv.threshold(gray_ref_img, 200, 255, cv.THRESH_BINARY)
    try:
        logging.debug('Aligning target image to reference image ...')
        aligned_img, overlap_ratio = alignImages(target_img, ref_img, ref_mask)
        logging.info(f'Finished alignmend of {image_path} with a overlap_ratio {overlap_ratio}!')
    except Exception as e:
        logging.error(f'Aligning image {image_path} to reference image failed! Cannot perform quantification!')
        return quantified_plate

    # Perform quantification on grayscale image
    aligned_gray_img = cv.cvtColor(aligned_img, cv.COLOR_BGR2GRAY)

    # Extract grids
    logging.debug('Extract grid cells ...')
    grids = extract_all_grids(aligned_gray_img, grid_positions)

    # Specify the quantification function
    quantify_point = sum

    # Iterate over the grid assuming structure [grid pos, cell row, cell column, cell data]
    logging.debug('Quantify grid cell content ...')
    quantgrids = np.zeros(shape=grids.shape[0:-2] + (1,))
    shape = grids.shape
    for grid_pos in range(shape[0]):
        for row in range(shape[1]):
            for column in range(shape[2]):
                quantgrids[grid_pos, row, column] = quantify_point(grids[grid_pos, row, column, :, :])

    # Generate an annotated image
    annotated = generate_annotated_image(aligned_img, quantgrids)
    annotated_dir = os.path.join(os.path.dirname(image_path), 'annotated_images')
    os.makedirs(annotated_dir, exist_ok=True)
    a_img_path = os.path.join(annotated_dir, '%s_Labeled.png' % os.path.basename(image_path))
    logging.info('Writing annotated image to %s ...', a_img_path)
    cv.imwrite(a_img_path, annotated)

    # Transform the numpy array into a pandas.DataFrame
    shape = quantgrids.shape
    cells = np.repeat(range(shape[0]), shape[1]*shape[2])
    rows = np.tile(np.repeat(range(shape[1]), shape[2]), shape[0])
    columns = np.tile(np.tile(range(shape[2]), shape[1]), shape[0])

    quantified_plate = pd.DataFrame({'plate_id': img_id, 'plate_image_file': os.path.basename(image_path),  'cell': cells, 'row': rows, 'column': columns, 'point':quantgrids.ravel()})

    return quantified_plate


@cfy
def generate_annotated_image(img, quantified_grid, zoom_factor=4, grid_size=48, grid_positions=[]):
    big_img = cv.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv.INTER_CUBIC)
    annotated_image = label_intensities(big_img, 
                            grid_positions =[(x*zoom_factor ,y*zoom_factor) for x,y in grid_positions], 
                            values = quantified_grid,
                            grid_size = zoom_factor*grid_size,
                            #strformat = '%2.2f'
                            strformat = '%d'
                           )
    return annotated_image


@cfy
def label_intensities(img, grid_positions, values, grid_size=48, nrows=4, ncols=4, strformat='%.1f', grid_color=(0, 255, 0), text_color=(255, 0, 0)):
    labeled = img.copy()
    for grid_pos,(x,y) in enumerate(grid_positions):
        cv.rectangle(labeled, (x, y), (x + grid_size, y + grid_size), (0, 255, 0), 1)
        for row in range(1, nrows):
            cv.line(labeled, (x, y + grid_size//nrows * row), (x + grid_size, y + grid_size//nrows * row), grid_color, 1)
        for col in range(1, ncols):
            cv.line(labeled, (x + grid_size//nrows * col, y), (x + grid_size//nrows * col, y + grid_size), grid_color, 1)
        for row in range(nrows):
            for col in range(ncols):
                tX = x + grid_size//nrows * col 
                tY = y + grid_size//nrows * (row + 1) - 1
                value = strformat % (values[grid_pos, row, col])

                cv.putText(labeled, value, (tX, tY), cv.FONT_HERSHEY_SIMPLEX, 0.3, text_color)

    return labeled


@cfy
def get_grid_elements(roi, ncols=4, nrows=4):
    rows = np.split(roi, nrows, axis=0)
    cells = [np.split(row, ncols, axis=1) for row in rows]
    return np.stack(cells)


@cfy
def extract_all_grids(image, grid_pos, grid_size=48):
    grids = []
    for x, y in grid_pos:
        grids.append(get_grid_elements(image[y:y+grid_size, x:x+grid_size]))
    return np.stack(grids)


def get_top_image(image, top_ratio=0.3):
    # Inveret ratio, taking only the top
    top_ratio = min(1.0, 1.0-top_ratio)
    
    inv_img = (255 - image).ravel()
    inv_img.sort()
    
    top_elements = int(len(inv_img)*top_ratio)
    return inv_img[top_elements:]


def mean(image):
    return np.mean(255 - image)
    #return np.sum(255 - image)

def sum(image):
    return np.sum(255 - image)

    
def top_mean(image, top_ratio=0.5):
    """
    Calculate the mean intensity of the darkest 50% of the image
    """
    return np.mean(get_top_image(image, top_ratio=top_ratio))

def top_sum(image, top_ratio=5.0):
    """
    Calculate the mean intensity of the darkest 50% of the image
    """
    return np.sum(get_top_image(image, top_ratio=top_ratio))


def top_median(image, top_ratio=5.0):
    """
    Calculate the mean intensity of the darkest 50% of the image
    """
    return np.median(get_top_image(image, top_ratio=top_ratio))


@cfy
def alignImages(target_img, ref_img, ref_mask, alignment_overlap_threshold=0.95, alignment_hyper_parameters=[(500, 0.10), (500, 0.05)]):
    """Perform an multiple image alignments, returning the best
    
    Arguments:
        target_img {np.array} -- Color target image that needs alignment
        ref_img {np.array} -- Reference image used as a template
        ref_mask {np.array} -- Binary masked of the reference image
    
    Keyword Arguments:
        alignment_overlap_threshold {float} -- Minimum overlap ratio between reference image mask, and aligned image mask (default: {0.95})
        alignment_hyper_parameters {list} -- Set of hyper parameters for alignment function (default: {[(500, 0.10), (500, 0.05)]})
    
    Returns:
        [np.array] -- Aligned image

    Raises:
        ValueError -- If no good enough alignmend is found
    """
    aligned_images = []
    overlap_ratios = []
    for max_features, good_match_percent in (alignment_hyper_parameters):
        aligned_img, _ = homography_alignment(target_img, ref_img, max_features, good_match_percent)

        gray_aligned_img = cv.cvtColor(aligned_img,cv.COLOR_BGR2GRAY) 
        _, aligned_mask = cv.threshold(gray_aligned_img, 200, 255, cv.THRESH_BINARY)

        # Calculate how well the aligned image mask equals to the reference mask
        overlap = ref_mask == aligned_mask
        overlap_ratio = overlap.sum() / overlap.size

        aligned_images.append(aligned_img)
        overlap_ratios.append(overlap_ratio)

    order = np.argsort(overlap_ratios)
    best = overlap_ratios[order[-1]]
    if best >= alignment_overlap_threshold:
        logging.debug(f'Best alignment image with an overlap_ratio {best} ...')
        return aligned_images[order[-1]], best
    else:
        logging.warn(f'Alignment failed! overlap_ratio: {best}')

    raise ValueError('No alignmend found!')


def homography_alignment(im1, im2, alignment_max_features, alignment_good_match_percent):
    """Perform an homographic alignment of im1 toward im2
    Code taken from: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

    Arguments:
        im1 {np.array} -- Target image
        im2 {np.array} -- Reference image
        alignment_max_features {int} -- Number of features to extract using ORB
        alignment_good_match_percent {float} -- Ratio of best features to keep for alignment
    
    Returns:
        [np.array] -- Aligned image
        [np.array] -- Image showing matches between unaligned target image and reference image
    """
    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv.ORB_create(alignment_max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * alignment_good_match_percent)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv.warpPerspective(im1, h, (width, height))

    return im1Reg, imMatches


# This should go into a separate module at some point
def multiprocess_tracking(func, iter, args=(), kwargs={}, tqdm_kwargs={}, single_process=False, max_workers=None, expand_args=False):
    """Using ProcessPoolExecutor to run *func* on multiple processes
    
    Arguments:
        func {function} -- Function to be executed
        iter {iterable} -- Call function on each of the elements
    
    Keyword Arguments:
        args {tuple} -- Arguments to pass to func (default: {()})
        kwargs {dict} -- kwargs passed to func (default: {{}})
        tqdm_kwargs {dict} -- kwargs passed to tqdm (default: {{}})
        single_process {bool} -- If set *True* will run in a single process (usefull for debugging) (default: {False})
        max_workers {[type]} -- Maximum number of workers to be used (default: {None})
        expand_args {[bool]} -- If set *True* expand arguments by calling func(*arg) on each
    
    Returns:
        [type] -- A list of results returned by func
    """
    # Store the parts here
    parts = []

    if single_process:
        for i in iter:
            if expand_args:
                part = func(*i, *args, **kwargs)
            else:
                part = func(i, *args, **kwargs)
            parts.append(part)
    else:
        # Multi process
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in iter:
                if expand_args:
                    future = executor.submit(func, *i, *args, **kwargs)
                else:
                    future = executor.submit(func, i, *args, **kwargs)
                futures.append(future)
            
            for finished_future in tqdm(as_completed(futures), **tqdm_kwargs):
                parts.append(finished_future.result())
    return parts


def get_boundingbox(contour):
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    box_area = rect[1][0]*rect[1][1]
    return box, box_area, rect

# Extract bounding box
@cfy
def get_outter_bounding_box(mask, bounding_box_kernel_size=10):
    kernel = np.ones((bounding_box_kernel_size, bounding_box_kernel_size), np.uint8)
    
    # Apply close & open in order to get rid of small particles and holes
    complete_mask = cv.morphologyEx(cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel), cv.MORPH_OPEN, kernel)

    _ ,contours, hierarchy = cv.findContours(complete_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bb, bb_area, rect = get_boundingbox(contours[0])
    return bb




