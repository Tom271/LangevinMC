clear all
close all
lambda=logspace(0,3,2);
 
T=100000;
dt=0.01;
x(:,1)=[1;1];
y(:,1)=x(:,1);
t=[0:dt:T];
A=diag(lambda);
%coefficent for the exact numerical scheme
a(1)=exp(-lambda(1)*dt); 
a(2)=exp(-lambda(2)*dt);
A1=diag(a);
b(1)=sqrt(1/lambda(1)*(1-a(1)^2));
b(2)=sqrt(1/lambda(2)*(1-a(2)^2));
 
 
 
for i=1:length(t)-1
    noise=randn(2,1);
    x(:,i+1)=x(:,i)-A*dt*x(:,i)./(1+dt*norm(A*x(:,i),2))+sqrt(2*dt)*noise;
    y(:,i+1)=A1*y(:,i)+[b(1)*noise(1);b(2)*noise(2)];
end
 
figure(1), histogram(x(1,:)), hold on, histogram(y(1,:))%plot(t,x(1,:),'r',t,y(1,:),'b--o')
figure(2), plot(x(1,:),x(2,:),'r',y(1,:),y(2,:),'b')