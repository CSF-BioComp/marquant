import cv2 as cv
import imutils
import numpy as np
import pandas as pd
import os
import logging

import glob
from concurrent.futures import ProcessPoolExecutor
import logging
from functools import partial
from tqdm import tqdm

# Extracted grid origin by hand
grid_positions = [
  [138, 114],
  [255, 114],
  [138, 256],
  [255, 256],
  [138, 398],
  [255, 398],
  [138, 540],
  [255, 540]
]


def slide_experiment(image_folder_path, single_process):
    imgfiles = glob.glob(os.path.join(image_folder_path, '**/*.tif'), recursive=True) 

    # Wrap in progress bar
    pack = zip(imgfiles, range(len(imgfiles))) 
    imgfiles = tqdm(imgfiles, desc='Image Quantification')

    logging.info('Found %s files ... ' % (len(imgfiles)))
    partial_results = []
    if single_process:
        for p in pack:
            partial_results.append(slide_quantifiaction_wrapper(p))
    else:
        with ProcessPoolExecutor(max_workers=None) as executor:
            it = executor.map(slide_quantifiaction_wrapper, pack)
        # Unpack
        partial_results = [x for x in it]

    # Join partial results and order table
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


def slide_quantifiaction(image_path, img_id, ref_image_path='reference_plate.png'):
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
    logging.info('Loading image ...')
    ref_img = cv.imread(ref_image_path)
    target_img = cv.imread(image_path)

    try:
        logging.debug('Aligning target image to reference image ...')
        aligned_img, matches = alignImages(target, ref, max_feats, good_matches)
    except Exception as e:
        logging.error('Aligning image to reference image failed! Cannot perform quantification!')
        return quantified_plate

    # Perform quantificaiton on grayscale image
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
    a_img_path = '%s_Labeled.png' % os.path.splitext(image_path)[0]
    logging.info('Writing annotated image to %s ...', a_img_path)
    cv.imwrite(a_img_path, annotated)

    # Transform the numpy array into a pandas.DataFrame
    shape = quantgrids.shape
    cells = np.repeat(range(shape[0]), shape[1]*shape[2])
    rows = np.tile(np.repeat(range(shape[1]), shape[2]), shape[0])
    columns = np.tile(np.tile(range(shape[2]), shape[1]), shape[0])

    quantified_plate = pd.DataFrame({'plate_id': img_id, 'plate_image_file': os.path.basename(image_path),  'cell': cells, 'row': rows, 'column': columns, 'point':quantgrids.ravel()})

    return quantified_plate


def generate_annotated_image(img, quantified_grid, zoom_factor=4, grid_size=48):
    big_img = cv.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv.INTER_CUBIC)
    annotated_image = label_intensities(big_img, 
                            grid_positions =[(x*zoom_factor ,y*zoom_factor) for x,y in grid_positions], 
                            values = quantified_grid,
                            grid_size = zoom_factor*grid_size,
                            #strformat = '%2.2f'
                            strformat = '%d'
                           )
    return annotated_image


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


def get_grid_elements(roi, ncols=4, nrows=4):
    rows = np.split(roi, nrows, axis=0)
    cells = [np.split(row, ncols, axis=1) for row in rows]
    return np.stack(cells)

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



def alignImages(im1, im2, max_features=500, good_match_percent=0.10):
  # Convert images to grayscale
  im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
  im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv.ORB_create(max_features)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * good_match_percent)
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

