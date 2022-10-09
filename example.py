#!/usr/bin/env python
# coding: UTF-8 
# Created by Cmoon

from Detector import Detector


class Tester:
    def __init__(self):
        """
        实例化Detector可传参数:
        weights: .pt模型路径
        imgsz: 图片大小
        conf_thres: 置信度阈值
        iou_thres: iou阈值
        """
        self.yolo = Detector()

    def test(self):
        """
        调用detect方法可传参数:
        device: 使用的设备(azure kinect/电脑摄像头)
        mode: “realtime”(实时检测，按q退出） / “find”(检测到指定物品在画面范围内退出) / other(检测到任意物体在范围内退出)
        range: 画面中心多大范围内的检测结果被采用
        nosave: 是否保存图片到本地
        find: 需要寻找的物体名称
        classes: 哪些物体可以被检测到,字符串名称间用','分割(例:传"bottle,person"则只会检测到bottle和person)
        运行返回:List of YoloResult(每个YoloResult代表一个物品)

        YoloResult数据类型
        YoloResult.name: 标签名称
        YoloResult.box: 矩形框左上右下xyxy坐标
        YoloResult.x: 矩形框中心点x坐标
        YoloResult.y: 矩形框中心点y坐标
        YoloResult.conf: 置信度
        YoloResult.distance: 中心点深度值（暂未接入）
        """
        results = self.yolo.detect()
        for item in results:
            print(item)


def main():
    tester = Tester()
    tester.test()


if __name__ == '__main__':
    main()
