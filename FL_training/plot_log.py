import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_training_loss():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="", help="")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', default = False,help='whether i.i.d or not')

    args = parser.parse_args()

    total_train_loss = []
    total_train_acc = []
    total_test_loss = []
    total_test_acc = []

    lines = open(args.path).readlines()
    for line in lines:
        parts = line.split(" || ")
        round = int(parts[0][-2:])
        training_loss = float(parts[1][-5:])
        test_loss = float(parts[2][-5:])
        test_accuracy = float(parts[3][-7:])
        print(round, training_loss, test_loss, test_accuracy)

        total_train_loss.append(training_loss)
        total_test_loss.append(test_loss)
        total_test_acc.append(test_accuracy)

    plt.figure()
    plt.plot(range(len(total_train_loss)), total_train_loss)
    plt.ylabel("training loss")
    plt.savefig('./save/fed_{}_{}_C{}_iid{}.png'.format(args.dataset, args.epochs, args.frac, args.iid))

    # plt.figure()
    # plt.plot(range(len(total_test_acc)), total_test_acc)
    # plt.gca().set_ylim((0.0, 100.0))
    # plt.ylabel("test accuracy")
    # plt.savefig('./save/fed_{}_{}_C{}_iid{}.png'.format(args.dataset, args.epochs, args.frac, args.iid))

if __name__ == "__main__":
    plot_training_loss()
