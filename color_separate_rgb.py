import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color
import math

NUMBER_OF_CLUSTERS = 20

class SetLoadingImage:
    def __init__(self, img_path):
        self.img = self.read_image(img_path)

    def read_image(self, img_path):
        try:
            read_img = cv2.imread(img_path)
            img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)
            return img
        except cv2.error:
            print("画像読み込みエラーのため、終了します")
            sys.exit()

    @property
    def return_img(self):
        return self.img

class KMeansAnalyzer:
    def __init__(self, img):
        self.img = img  
        self.number_of_cluster = NUMBER_OF_CLUSTERS

    # def find_optimal_k(self):
    #     data = self.img.reshape(-1, 3).astype(np.float32) 
    #     distortions = []

    #     for i in range(1, 11):
    #         kmeans = KMeans(n_clusters=i, n_init=10)
    #         kmeans.fit(data)
    #         o = kmeans.inertia_
    #         distortions.append(o)

    #     optimal_k = np.argmin(np.diff(distortions)) + 1
    #     return optimal_k

    def analyze(self):
        colors = self.img.reshape(-1, 3).astype(np.float32) 
        criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0

        _, labels, rgb_value = cv2.kmeans(
            data=colors, 
            K=self.number_of_cluster,  
            bestLabels=None, 
            criteria=criteria,  
            attempts=10, 
            flags=cv2.KMEANS_RANDOM_CENTERS, 
        )

        self.labels = labels.squeeze(axis=1)  # (N, 1) -> (N,)のように要素数が1の次元を除去する
        self.rgb_value = rgb_value.astype(np.uint8)  # float32 -> uint8

        _, self.counts = np.unique(
            self.labels, axis=0, return_counts=True
        )  # 重複したラベルを抽出し、カウント（NUMBER_OF_CLUSTERSの大きさだけラベルタイプが存在する）

        self.df = self.__summarize_result(self.rgb_value, self.counts)

        return self.df

    # 計算結果をグラフ用にDataFrame化させる
    @staticmethod
    def __summarize_result(rgb_value, counts):
        df = pd.DataFrame(data=counts, columns=["counts"])
        df["R"] = rgb_value[:, 0]
        df["G"] = rgb_value[:, 1]
        df["B"] = rgb_value[:, 2]

        # plt用に補正
        bar_color = rgb_value / 255
        df["plt_R_value"] = bar_color[:, 0]
        df["plt_G_value"] = bar_color[:, 1]
        df["plt_B_value"] = bar_color[:, 2]

        # グラフ描画用文字列
        bar_text = list(map(str, rgb_value))
        df["plt_text"] = bar_text

        # countsの個数順にソートして、indexを振り直す
        df = df.sort_values("counts", ascending=True).reset_index(drop=True)
        return df

class MakeFigure:
    def __init__(self, dataframe, rgb_value, labels):
        self.df = dataframe
        self.number_of_cluster = NUMBER_OF_CLUSTERS
        self.rgb_value = rgb_value
        self.labels = labels

    def output_histgram(self, ax):
        rgb_value_counts = (
            self.df.loc[:, ["counts"]].to_numpy().flatten().tolist()
        )  # ヒストグラム用のrgb値カウント数

        bar_color = (
            self.df.loc[:, ["plt_R_value", "plt_G_value", "plt_B_value"]]
            .to_numpy()
            .tolist()
        )  # ヒストグラム用のrgb値カウント数

        bar_text = self.df.loc[:, ["plt_text"]].to_numpy().flatten()  # ヒストグラム用x軸ラベル

        # ヒストグラムを表示する。
        ax.barh(
            np.arange(self.number_of_cluster),
            rgb_value_counts,
            color=bar_color,
            tick_label=bar_text,
        )

        ax.set_xlabel("Frequencies")
    
class Data:
    def __init__(self,df):
        self.colorFrequency = df.loc[:, ["counts"]].to_numpy().flatten().tolist()

    def rgb2cielab(self,rgbs):
        labs = []
        for rgb in rgbs:
            labs.append(color.rgb2lab(rgb.reshape(1, 1, 3)))
        return labs
    
    def average_similarity(self,labs):
        distances = []
        lab_values = np.array(labs)
        for i in range(len(lab_values)):
            for j in range(i+1, len(lab_values)):
                distance = np.linalg.norm(lab_values[i] - lab_values[j])
                distances.append(distance)
        average_distance = np.mean(distances)
        return average_distance

    # def getFrequency(self):
    #     print(f'Number of colors : {len(self.colorFrequency)}')
    #     average = np.average(self.colorFrequency)
    #     count = 0
    #     for num in self.colorFrequency:
    #         if num >= average:
    #             count+=1
    #     return count



