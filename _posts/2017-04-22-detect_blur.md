---
layout: post
title:  "python检测图片清晰图"
date:   2017-04-22 09:15:31
categories: python
tags: python
---

* content
{:toc}


参考[博客](http://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)


整理下代码来

```python
# -*- coding: UTF-8 -*-
import cv2
import argparse
import os


def list_images(base_path, contains=None):
    return list_files(base_path, valid_exts=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)


def list_files(base_path, valid_exts=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    # loop over the directory structure
    for (root_dir, dir_names, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(valid_exts):
                # construct the path to the image and yield it
                image_path = os.path.join(root_dir, filename).replace(" ", "\\ ")
                yield image_path


def get_result(images_path, output_file):
    results = []
    for image_path in list_images(images_path):
        # load the image, convert it to grayscale, and compute the
        # focus measure of the image using the Variance of Laplacian
        # method
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        file_path, filename = os.path.split(image_path)
        results.append("%s\t%s" % (filename, fm))

    with open(output_file, "w") as f:
        f.write("\n".join(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="path to input directory of images")
    ap.add_argument("-o", "--output", required=True, help="path to output filename of result")
    args = vars(ap.parse_args())
    images_path = args["images"]
    output_file = args["output"]
    get_result(images_path, output_file)


if __name__ == '__main__':
    main()

```

