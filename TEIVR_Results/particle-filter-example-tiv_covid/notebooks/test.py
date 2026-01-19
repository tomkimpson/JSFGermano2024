import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _():
    import scienceplots
    return


@app.cell
def _(plt):
    plt.style.use('science')
    return


@app.cell
def _(plt):
    def plot_function_1():
        plt.scatter(1,1)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        plt.show()
    return (plot_function_1,)


@app.cell
def _(plot_function_1):
    plot_function_1()
    return


@app.cell
def _():
    import corner
    return (corner,)


@app.cell
def _(corner):
    import numpy as np 
    def plot_function_2():
        np.random.seed(42)
        samples = np.random.randn(1000, 2)



        corner_kwargs = {
            'labels': [r"$\alpha$", r"$\beta$"],
            'color': 'teal',
            'bins': 30,
            'plot_datapoints': True,
            'plot_density': True,
            'plot_contours': True,
            'data_kwargs': {'alpha': 0.2, 'color': 'lightblue'},
            'hist_kwargs': {'alpha': 0.8, 'color': 'teal'},
            'contour_kwargs': {'colors': 'teal'},
            'smooth': 1,
            'smooth1d': 1,
            'quantiles': [0.16, 0.5, 0.84],
            'show_titles': True,
            'title_kwargs': {"fontsize": 20},
            'label_kwargs': {"fontsize": 22}
        }






        fig = corner.corner(samples,**corner_kwargs)
        return fig 
        #plt.show()
    return (plot_function_2,)


@app.cell
def _(plot_function_2):
    plot_function_2()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
