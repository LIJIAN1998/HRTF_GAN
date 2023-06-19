# # Secondly, try to use Bayesian optimization search to perform hyperparameter tuning
import GPyOpt
import GPy
import torch.nn as nn
import torch.optim as optim

# Initialise some dropout rate for tuning
rate_dropout = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
# Initialise my weight method
method_for_weight = [nn.init.kaiming_uniform_,
          nn.init.kaiming_normal_,
          nn.init.xavier_uniform_,
          nn.init.xavier_normal_]


# Choose which model to run test, Resnet in this cw
current_model = MyResNet(weight_method=method_for_weight[0], dropout=rate_dropout[2])
current_optimizer = optim.Adamax(current_model.parameters(), lr=0.0005, weight_decay=0.0000841)
# Calculate amount of parameters
# total_parameters = sum(p.numel() for p in current_model.parameters() if p.requires_grad)
# print("The amount of model parameters is: {}".format(total_parameters))
# Start train
# train_part(current_model, current_optimizer, epochs = 10)

# # Show final test accuracy
# check_accuracy(loader_val, current_model, analysis=True)

# # Finally, save current model
# torch.save(current_model.state_dict(), 'model.pt')


def Bayesian_Optimization_Search(hyperparameter):
    # Choose resnet to do tuning process
    current_model = MyResNet(weight_method=method_for_weight[0], dropout=rate_dropout[2])
    # Squeeze all hyperparameters
    current_hyperparameter = hyperparameter.squeeze()
    # Choose optimizer of model
    current_optimizer = optim.Adam(current_model.parameters(), lr=current_hyperparameter[0], weight_decay=current_hyperparameter[1])
    # Setup randomly train batch
    rand_num = int(torch.rand(1) * 40000)
    print("Randomly selected train batch: %d-%d Learning, Rate = %.6f, Weight Decay = %.6f" % (
    rand_num, rand_num + 1000, current_hyperparameter[0], current_hyperparameter[1])) 
    train_part(current_model, current_optimizer, epochs=10) 
    return check_accuracy(loader_val, current_model) 
# Initialise keys 
keys = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)}, 
{'name': 'weight_decay', 'type': 'continuous', 'domain': (1e-5, 1e-2)}]
# Start optimise parameters by using bayesian optimization search 
test_Bayesian_Optimization = GPyOpt.methods.BayesianOptimization(f=Bayesian_Optimization_Search, 
domain=keys, 
model_type='GP', 
acquisition_type='EI', 
maximize=True) 
test_Bayesian_Optimization.run_optimization(20, 60, 10e-6) 
# Plot result to better comparison improvement 
test_Bayesian_Optimization.plot_convergence() 
lr, wd = test_Bayesian_Optimization.x_opt 
print("Obtain! Current best Hyperparameter with lr = %.7f, Weight Decay = %.7f, obtained accuracy: %.2f" % (  lr, wd, test_Bayesian_Optimization.fx_opt))