import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_loss():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="", help="")
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

    # plt.figure()
    # plt.plot(range(len(total_train_loss)), total_train_loss)
    # plt.ylabel("training loss")
    # plt.savefig("training_loss.png")

    plt.figure()
    plt.plot(range(len(total_test_acc)), total_test_acc)
    plt.gca().set_ylim((0.0, 100.0))
    plt.ylabel("testing accuracy")
    plt.savefig("./testing_accuracy.png")


if __name__ == "__main__":
    plot_training_loss()
