% Broyden's Method for solving Backward Euler

clear all
close all
options = optimset('Display','off');
% discretize grid in space
Nx=181; %need odd number of points
x=linspace(0,2*pi,Nx+1)';
dx=x(2)-x(1);
x(Nx+1)=[];
%define Fourier matrices
Iden=eye(Nx);
K=diag(-(Nx-1)/2:(Nx-1)/2);
Dx=ifft(ifftshift(1i*K*fftshift(fft(Iden))));  % Differentiation Matrix using fourier transforms

% Define conditions for Euler's Method
uinit = sin(x);
Tfinal = 1.75;
Nt=100;
dt = Tfinal/Nt;
un = uinit;

% Residual Tolerance for Broyden's Method to break
 ResTol = 1e-14;
 tic
 for i = 1:Nt
     fun = @(y) (y - un + dt*Dx*(0.5*y.^2)); % Backward Euler's Equation
     u = un;
     J = Iden + dt * Dx * diag(u);
     Bi= inv(J);
     for ia = 1:50  % Broyden's loop
         fun_old = fun(u);
         s = -Bi* fun_old;
         u = u + s;
         if(norm(fun(u),inf)<ResTol)
            break;
         end
         y = fun(u) - fun_old;     
         % Bi= Bi+ ((s - Bi*y)*y')/(norm(y)^2); %update the inverse, Bad Broyden's Method
         bool = true;
         Bi= Bi+ ((s - Bi*y)* s'*Bi)/(s'*Bi*y); % Good Broyden's Method
     end

     unp1=u;
     un = unp1;
     tElapsed = toc;
     plot(x,un)
     pause(0.01)
     tic;
 end
 tElapsed = toc

 uStar = u
 final = fun(uStar)