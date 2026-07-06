function pde = A120data
%% SINCOSDATA trigonometric  data for Poisson equation
%
%     f = 2*pi^2*cos(pi*x)*cos(pi*y);
%     u = cos(pi*x)*cos(pi*y);
%     Du = (-pi*sin(pi*x)*cos(pi*y), -pi*cos(pi*x)*sin(pi*y));
%
% The u satisfies the zero flux condition du/dn = 0 on boundary of [0,1]^2
% and thus g_N is not assigned.
%
% Copyright (C)  Long Chen. See COPYRIGHT.txt for details.

pde = struct('f',@f,'exactu',@exactu,'g_D',@g_D,'Du',@Du);


    % load data (right hand side function)
    function rhs =  f(p)
    S = 120;
    x = p(:,1); y = p(:,2);
    rhs = 2.*y.*((16.*atan(S.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16)))./pi - 8)...
    .*(y - 1) + 2.*x.*((16.*atan(S.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16)))...
    ./pi - 8).*(x - 1) + (64.*S.*x.*y.*(x - 1).*(y - 1))./(pi.*(S.^2.*((x...
    - 1./2).^2 + (y - 1./2).^2 - 1./16).^2 + 1)) + (32.*S.*x.*y.*(2.*x -...
    1).*(y - 1))./(pi.*(S.^2.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16).^2 ...
    + 1)) + (32.*S.*x.*y.*(2.*y - 1).*(x - 1))./(pi.*(S.^2.*((x - 1./2).^2 ...
    + (y - 1./2).^2 - 1./16).^2 + 1)) + (32.*S.*x.*(2.*y - 1).*(x - 1).*(y - ...
    1))./(pi.*(S.^2.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16).^2 + 1)) +...
    (32.*S.*y.*(2.*x - 1).*(x - 1).*(y - 1))./(pi.*(S.^2.*((x - 1./2).^2 + ...
    (y - 1./2).^2 - 1./16).^2 + 1)) - (32.*S.^3.*x.*y.*(2.*x - 1).^2.* ...
    (x - 1).*(y - 1).*((x - 1./2).^2 + (y - 1./2).^2 - 1./16))./(pi.* ...
    (S.^2.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16).^2 + 1).^2) - (32.* ...
    S.^3.*x.*y.*(2.*y - 1).^2.*(x - 1).*(y - 1).*((x - 1./2).^2 + (y - ...
    1./2).^2 - 1./16))./(pi.*(S.^2.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16).^2 + 1).^2);
    end
    % exact solution
    function u =  exactu(p)
    S = 120;
    x = p(:,1); y = p(:,2);
    u = 16 * (0.5 + atan(S * (0.25 ^ 2 - (x - 0.5) .^ 2 - (y - 0.5) .^ 2)) / pi) .* x .* (x - 1) .* y .* (y - 1);
    end
    % Dirichlet boundary condition
    function u =  g_D(p)
    u =  exactu(p);
    end
    % Derivative of the exact solution
    function uprime =  Du(p)
    S = 120;
    x = p(:,1); y = p(:,2);
    uprime(:,1) = - x.*y.*((16.*atan(S.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16)))./pi - 8).*(y - 1) - y.*((16.*atan(S.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16)))./pi - 8).*(x - 1).*(y - 1) - (16.*S.*x.*y.*(2.*x - 1).*(x - 1).*(y - 1))./(pi.*(S.^2.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16).^2 + 1));
    uprime(:,2) = - x.*y.*((16.*atan(S.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16)))./pi - 8).*(x - 1) - x.*((16.*atan(S.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16)))./pi - 8).*(x - 1).*(y - 1) - (16.*S.*x.*y.*(2.*y - 1).*(x - 1).*(y - 1))./(pi.*(S.^2.*((x - 1./2).^2 + (y - 1./2).^2 - 1./16).^2 + 1));
    end
end