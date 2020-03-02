# -*- coding: utf-8 -*-
'''
 @File  : get_weibo_emoji.py
 @Author: ChangSiteng
 @Date  : 2020-02-21
 @Desc  : 
 '''

# import-path
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import requests




if __name__ == '__main__':


    url = "https://api.weibo.com/2/emotions.json?access_token=2.00oe5MIGcrCVTD2180f1021cJXiKKD"
    response = requests.get(url).json()
    for res in response:
        emoji_name = str(res['value']).replace('[','').replace(']','')
        emoji_url = res['url']
        r = requests.get(emoji_url, stream=True)
        if r.status_code == 200:
            open('./emojis/'+emoji_name+'.png', 'wb').write(r.content)  # 将内容写入图片
        else:
            print(r,",---",emoji_name)
        del r
    print("done!!!!!!!")
