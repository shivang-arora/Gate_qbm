function phi_proc = phaseangles_qbm(tau, delta, deg)
    warning("off", "all")
    targ = @(x) exp(-tau * x);

    opts.intervals=[delta,1];
    opts.objnorm = 2;
    % opts.epsil is usually chosen such that target function is bounded by 1-opts.epsil over D_delta 
    opts.epsil = 0.2;
    opts.npts = 500;
    opts.fscale = 1;
    opts.isplot = false;
    coef_full=cvx_poly_coef(targ, deg, opts);

    parity = mod(deg, 2);
    coef = coef_full(1+parity:2:end);

    opts.maxiter = 100;
    opts.criteria = 1e-12;
    opts.useReal = false;
    opts.targetPre = true;
    opts.method = 'Newton';
    % solve phase factors
    [phi_proc,out] = QSP_solver(coef,parity,opts);

    %xlist = linspace(delta,1,500)';
    %func = @(x) ChebyCoef2Func(x, coef, parity, true);
    %func_value = func(xlist);
    %QSP_value = QSPGetEntry(xlist, phi_proc, out);
    %err= norm(QSP_value-func_value,Inf);
    %disp('The residual error is');
    %disp(err);
end