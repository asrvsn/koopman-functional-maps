import matplotlib.pyplot as plt

def plot_images(d: dict):
	fig = plt.figure(figsize=(3*len(d.keys()), 3))
	for i, (title, img) in enumerate(d.items()):
		ax = fig.add_subplot(1, 4, i+1)
		ax.imshow(img, interpolation='nearest')
		ax.set_title(title, fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])
	fig.tight_layout()
	return fig
