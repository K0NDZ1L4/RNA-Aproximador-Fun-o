import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor





n_file=5
sim = [[3, 3,10000], [10, 5,1000], [30, 15,200]]




def Teste_3_simulações(n_file,sim):
    print('Carregando Arquivo de teste')
    arquivo = np.load('teste'+str(n_file)+'.npy')
    x = arquivo[0]
    y = np.ravel(arquivo[1])

    for z in range(len(sim)):
        erro = []
        erro_temp = 10000000000000000000
        for zz in range(10):
            regr = MLPRegressor(hidden_layer_sizes=(sim[z][0], sim[z][1]),
                                max_iter=sim[z][2],
                                activation='relu',  # {'identity', 'logistic', 'tanh', 'relu'},
                                solver='adam',
                                learning_rate='adaptive',
                                n_iter_no_change=50)
            print('Treinando RNA')
            regr = regr.fit(x, y)
            y_est = regr.predict(x)
            erro.append(regr.best_loss_)
            if(regr.best_loss_ < erro_temp):
                erro_temp = regr.best_loss_
                regr_temp = regr
                Y_est_temp = y_est



        plt.figure(figsize=[14, 7])
        # plot curso original
        plt.subplot(1, 3, 1)
        plt.plot(x, y)
        # plot aprendizagem
        plt.subplot(1, 3, 2)
        plt.plot(regr_temp.loss_curve_)
        # plot regressor
        plt.subplot(1, 3, 3)
        plt.plot(x, y, linewidth=1, color='yellow')
        plt.plot(x, Y_est_temp, linewidth=2)
        # Mostrar gráficos
        plt.show()

        media = sum(erro)/len(erro)
        desvio_padrao = np.std(erro)
        print("Média Simulação {0}:{1}".format(z+1, media))
        print("Desvio Padrão {0}:{1}".format(z+1, desvio_padrao))



Teste_3_simulações(n_file,sim)