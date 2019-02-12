import pickle
import matplotlib.pyplot as plt

epsilon = [0.02, 0.05, 0.1, 0.2, 0.5]
dalalyan = [(725435.0, 2.3481638532950921e-05), (71502.0, 0.00018697916781715065), (11235.0, 0.00094325075818563187), (1535.0, 0.0050994091499003621), (72.0, 0.058358268686440291)]
plt.scatter(range(1, len(epsilon)+1), epsilon)
plt.legend(['Sought total variation epsilon'], loc='upper left')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.ylabel('Total variation')
for i, eps in enumerate(epsilon, 1):
    plt.annotate( str('eps={:.2f}\nN={:.0f}\nh={:.5f}'.format(epsilon[i-1], *dalalyan[i-1])) , (i+0.15 if i != len(epsilon) else i-.8, eps- (.02 if i != len(epsilon) else .04) ))

results = []
for N, step in dalalyan:
    N = int(N)
    e = pickle.load(open( 'N_' + str(N) + '_step_' + str(step), 'rb' ))
    print(e["step"], e["N"], e["results"])
    results.append(e["results"][0])
plt.boxplot(results)

plt.show()
