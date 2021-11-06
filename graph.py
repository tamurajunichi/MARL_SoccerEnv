import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from numba import jit
import os
import glob

TITLE_LOOKUP = {
    0:"Episode Rewards",
    1:"Intrinsic Episode Rewards",
    2:"Current Q-Values",
    3:"Mixed Q-Values",
    4:"Critic Network Loss",
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
        y = debias_smooth(y, weight=0.9999)
        sub_fig.plot(x,y)
        sub_fig.set_title(TITLE_LOOKUP[i])
        sub_fig.ticklabel_format(style="sci", axis="x",scilimits=(0,0))
        sub_fig.xaxis.set_major_locator(MaxNLocator(11))
    figure.subplots_adjust(wspace=0.4, hspace=0.6)

    plt.savefig("./log_figure/{}.jpg".format('.'.join(file_name.split('.')[:2])))
    # plt.show()

def main():
    seed = "572682235"
    path = "./agent_log/"
    file = glob.glob(path + "*_{}_*".format(seed))
    for f in file:
        if not "timestep" in f:
            file = f
            break
    output_graph(file.split('/')[-1])

if __name__ == '__main__':
    main()