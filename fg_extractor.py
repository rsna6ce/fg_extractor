#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import os
import sys
import shutil
import subprocess
import argparse
import cv2
import numpy as np
import glob

block_size = 8

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__))
    parser.add_argument('-i', '--input', help='input mp4 file', required=True, dest='input')
    parser.add_argument('-b', '--bg_image', help='filename of fg image', required=True, type=str, dest='bg_image')
    parser.add_argument('-o', '--output', help='output mp4 file. default=(input).mp4', default='', dest='output')
    parser.add_argument('-f', '--frame_rate', help='frame_rate fps', default=30, dest='frame_rate')
    parser.add_argument('-d', '--diff_threshold', help='diff threshold of fg-bg', default=32, type=int, dest='diff_threshold')
    parser.add_argument('-r', '--roi_offset', help='roi expand offset', default=8, type=int, dest='roi_offset')
    parser.add_argument('-s', '--start_frame', help='start frame number', default=0, type=int, dest='start_frame')
    parser.add_argument('-e', '--end_frame', help='end frame number', default=0, type=int, dest='end_frame')
    parser.add_argument('-n', '--name', help='name of object', default='', type=str, dest='name')
    parser.add_argument('-t' '--triming', help='trim: paint black outside of roi', action='store_true', dest='triming')
    args = parser.parse_args()
    fg_extractor(args)

def fg_extractor(param):
    # clean up temp dir
    script_dir = os.path.dirname(__file__)
    temp_dir = script_dir + '/temp'
    for filename in  glob.glob(temp_dir + '/*.png'):
        os.remove(filename)

    #load bg image
    img_bg = cv2.imread(param.bg_image)
    height, width, ch = img_bg.shape
    width2 = int(width/block_size)
    height2 = int(height/block_size)
    img_bg2 = cv2.resize(img_bg, (width2, height2), interpolation=cv2.INTER_AREA)
    img_bg2 = cv2.cvtColor(img_bg2, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(param.input)
    frame_number = 0
    frame_count = 0
    while True:
        ret, img_fg = cap.read()
        if not ret:
            break

        if (frame_number < param.start_frame) or \
           (0 < param.end_frame and param.end_frame < frame_number):
            print('frame_number:', frame_number, '(skip)')
            frame_number+=1
            continue

        img_bk = np.full((height, width, 3), (0, 255, 0), dtype = np.uint8)

        img_fg2 = cv2.resize(img_fg, (width2, height2), interpolation=cv2.INTER_AREA)
        img_fg2 = cv2.cvtColor(img_fg2, cv2.COLOR_BGR2GRAY)

        img_diff = cv2.absdiff(img_bg2, img_fg2)
        img_diff = cv2.resize(img_diff, (width, height), interpolation=cv2.INTER_NEAREST)
        _, img_th = cv2.threshold(img_diff, param.diff_threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours):
            min_x = width
            min_y = height
            max_x = 0
            max_y = 0
            roi_offset = param.roi_offset
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x+w)
                max_y = max(max_y, y+h)
            cv2.rectangle(img_fg, 
                (max(0, min_x-roi_offset), max(0, min_y-roi_offset)),
                (min(width-1, max_x+roi_offset), min(height-1, max_y+roi_offset)),
                (0, 0, 255), 2)
            # put name text
            if param.name:
                    rect_w = 150
                    rect_h = 20
                    cv2.rectangle(img_fg,
                        (min_x-roi_offset, min_y-roi_offset),
                        (min_x+rect_w+roi_offset, min_y+rect_h),
                        (0, 0, 255), cv2.FILLED, cv2.LINE_AA)

                    cv2.putText(img_fg,
                        text=param.name,
                        org=(min_x, min_y+roi_offset*2),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_4,
                        bottomLeftOrigin=False)
            # crop
            if param.triming:
                left, right, top, bottom = [
                    max(0, min_x-roi_offset), min(width-1, max_x+roi_offset), 
                    max(0, min_y-roi_offset), min(height-1, max_y+roi_offset)]
                #print(left, right, top, bottom)
                img_crop = img_fg[top:bottom, left:right]
                img_bk[top:bottom, left:right] = img_crop



        frame_number_with_zero = str(frame_count).zfill(8)
        temp_filename = temp_dir + '/' + frame_number_with_zero + '.png'
        if param.triming:
            cv2.imwrite(temp_filename, img_bk)
        else:
            cv2.imwrite(temp_filename, img_fg)
        print('frame_number:', frame_number, '('+frame_number_with_zero+'.png)')
        frame_number += 1
        frame_count += 1

    # encode with ffmpeg
    print('encoding with ffmpeg ....')
    output_mp4 = param.output if param.output!='' else param.input+'_out.mp4'
    frames_dir = temp_dir
    frame_rate = param.frame_rate
    subprocess.run(('ffmpeg' ,
        '-loglevel', 'warning',
        '-y',
        '-framerate', str(frame_rate),
        '-i', temp_dir+'/%8d.png',
        '-vframes', str(frame_count),
        '-vf', 'scale={0}:{1},format=yuv420p'.format(width, height),
        '-vcodec', 'libx264',
        '-r', str(frame_rate),
        output_mp4))

if __name__ == '__main__':
    sys.exit(main())
