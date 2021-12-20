import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
import numpy as np
from numba import jit
import os
import glob

TITLE_LOOKUP = {
    0:"Episode Rewards",
    1:"Intrinsic Episode Rewards",
    2:"current Q-Values",
    3:"ext Mixed Q-Values",
    4:"Total Critic Network Loss",
    5:"Predictor Network Loss",
    6:"Kick Count",
    7:"n_step"
}

@jit(nopython=True)
def ema(x, alpha):
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    return y

def debias_smooth(scalars,weight):
    last=scalars[0]
    last=0
    smoothed=[]
    for i,point in enumerate(scalars,start=1):
        smoothed_val=last*weight+(1-weight)*point
        last=smoothed_val

        debias_weight=1-weight**i
        smoothed_val=smoothed_val/debias_weight
        smoothed.append(smoothed_val)
    return smoothed

def output_graph(file_name):
    # agent_logからnpyのロード
    try:
        data = np.load('./agent_log/{}'.format(file_name))
    except Exception as e:
        print(file_name+" : "+str(e))
        return

    # ロードしたndarrayから一気に複数グラフの作成
    x = np.arange(data.shape[0])

    figure = plt.figure(figsize=(16,9))
    sub_figures = []
    # 3x2の6つでグラフ作成しリストに入れておく
    for i in range(7):
        sub_figures.append(figure.add_subplot(4,2,i+1))

    # サブグラフの設定
    for i, sub_fig in enumerate(sub_figures):
        y = data[:, i]
        y = debias_smooth(y, weight=0.99999)
        sub_fig.plot(x,y)
        sub_fig.set_title(TITLE_LOOKUP[i])
        sub_fig.ticklabel_format(style="sci", axis="x",scilimits=(0,0))
        sub_fig.xaxis.set_major_locator(MaxNLocator(11))
    figure.subplots_adjust(wspace=0.4, hspace=0.6)

    plt.savefig("./log_figure/{}.jpg".format('.'.join(file_name.split('.')[:2])))

def output_graph_overlap(file_names, title, w=0.9999, log_scale=False):
    color_list = ["r", "g", "b", "c", "m", "k", "w"]
    color_list = list(matplotlib.colors.BASE_COLORS.items())

    # 3x2の6つでグラフ作成しリストに入れておく
    figure = plt.figure(figsize=(16, 9))
    sub_figures = []

    # 表示させたいデータに合わせて変更
    for i in range(1):
        sub_figures.append(figure.add_subplot(3, 2, i + 1))

    for j,file_name in enumerate(file_names):
        data = np.load('./agent_log/{}'.format(file_name))
        # ロードしたndarrayから一気に複数グラフの作成
        x = np.arange(data.shape[0])

        # サブグラフの設定
        for i, sub_fig in enumerate(sub_figures):
            y = data[:, i]
            y = debias_smooth(y, weight=w)
            sub_fig.plot(x, y, label='.'.join(file_name.split('_')[-1].split('.')[:-1]), color=color_list[j][1])
            sub_fig.set_title(TITLE_LOOKUP[i])
            sub_fig.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            sub_fig.xaxis.set_major_locator(MaxNLocator(11))
            sub_fig.grid(b=True, which='major', color='gray', linestyle='-')
            if log_scale:
                sub_fig.set_yscale('log')
            sub_fig.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0)
            figure.subplots_adjust(wspace=0.4, hspace=0.6)

    # dataリスト数分処理
    plt.savefig("./log_figure/{}.jpg".format(title))

def main():
    #seeds = ["31455996", "141897654", "585355530", "74531053", "353498533"]
    #seeds= ["267673820","522963869","153365176"]
    # td3
    #seeds = ["289045825"]
    # RND lr low
    #seeds= ["248941593","123428569","282010902","6193399","278602160"]
    # epsilon 5
   # seeds= ["30323477","437636585","184751358","64906974","393951905"]
    # epsilon td3 false
    #seeds = ["12740750", "146821019", "681873539", "764377622", "876236796"]
    # epsilon td3 true
    #seeds = ["627660174","59806259","64600305","120225967","207008214","795455745","701079192"]
    # rnd td3 true lr 1e-4
    #seeds = ["939288846","496901005","270809199","963309437","634282329"]
    # rnd td3 true onli kick goal reward 50000
    #seeds = ["110720989","536059742","215788368","590761916","793603361"]
    # rnd td3 true onli kick goal reward 200000
    # seeds = ["101461999","530090364"]
    # rnd td3 true lr 1e-3
    # seeds = ["855443973", "648983353", "381305764", "60646118", "330199991"]
    # rnd td3 true lr 1e-3
    seeds = ["143214102","676938695","697006494","949112626","638942401"]
    path = "./agent_log/"
    files = []
    for seed in seeds:
        file = glob.glob(path + "*_{}_*".format(seed))
        for f in file:
            if not "timestep" in f:
                file = f
                file_name = file.split('/')[-1]
                files.append(file_name)
    output_graph_overlap(file_names=files, title="RND")

if __name__ == '__main__':
    main()
