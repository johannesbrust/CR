function [pars] = adampars()
%ADAMPARS Parameters for Adam

pars.alp                        = 0.001;             % learning rate
pars.bet1                       = 0.9;               % exp decay mean
pars.bet2                       = 0.999;             % exp decay var
pars.eps                        = 1e-8;              % perturbation

end