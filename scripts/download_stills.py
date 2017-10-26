# coding=utf-8
from __future__ import absolute_import

import os
import requests
import simplejson as json

from PIL import Image


def download():

    for i in range(10082, 15000):

        url = 'http://movie.mtime.com/{}/posters_and_images/stills/hot.html'.format(i)

        page = requests.get(url)
        content = page.content
        start_index = content.find('var imageList = ')
        end_index = content.find(',{"specialimages')
        json_str = content[start_index+16: end_index]+']'
        try:
            image_list = json.loads(json_str)
        except Exception:
            continue
        stills = image_list[0]
        stage_pictures = stills.get('stagepicture')
        stage_picture = stage_pictures[0]
        official_stage_images = stage_picture.get('officialstageimage')

        for k, official_stage_image in enumerate(official_stage_images):
            image_url = official_stage_image.get('img_1000')
            print image_url
            image = requests.get(image_url)
            with open('images/{}_{}.jpg'.format(i, k), 'wb+') as img:
                img.write(image.content)
            print 'images/{}_{}.jpg    Done!!!!'.format(i, k)


def filter():
    images = [f for f in os.listdir('images') if f != '.DS_Store']
    for i, image in enumerate(images):
        try:
            img = Image.open('images/{}'.format(image))
        except IOError:
            os.remove('images/{}'.format(image))
            print 'IOERROR'
            continue
        if img.height < 200 or img.width < 960:
            os.remove('images/{}'.format(image))
        print i

filter()
