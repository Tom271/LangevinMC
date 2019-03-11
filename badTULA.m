clear all
close all
lambda=[1,1000];
 
T=1000;
h=1;
x(:,1)=[1;1];
y(:,1)=x(:,1);
xc(:,1)=x(:,1);
xT=xc;
t=[0:h:T];
A=diag(lambda);
%coefficent for the exact numerical scheme
a(1)=exp(-lambda(1)*h); 
a(2)=exp(-lambda(2)*h);
A1=diag(a);

b(1)=sqrt(1/lambda(1)*(1-a(1)^2));
b(2)=sqrt(1/lambda(2)*(1-a(2)^2));
 
 
 
for i=1:length(t)-1
    noise=randn(2,1);
    %tULA 
    x(:,i+1)=x(:,i)-A*h*x(:,i)./(1+h*norm(A*x(:,i),2))+sqrt(2*h)*noise;
    %tULAc 
    xc(1,i+1)=xc(1,i)-A(1,1)*h*xc(1,i)./(1+h*norm(A(1,1)*xc(1,i),2))+sqrt(2*h).*noise(1);
    xc(2,i+1)=xc(2,i)-A(2,2)*h*xc(2,i)./(1+h*norm(A(2,2)*xc(2,i),2))+sqrt(2*h).*noise(2);
    
    %ULTAc 
    xT(1,i+1)=xT(1,i)-A(1,1)*h*xT(1,i)./max(1,h*norm(A(1,1)*xT(1,i),2))+sqrt(2*h).*noise(1);
    xT(2,i+1)=xT(2,i)-A(2,2)*h*xT(2,i)./max(1,h*norm(A(2,2)*xT(2,i),2))+sqrt(2*h).*noise(2);
    
    %True solution
    y(:,i+1)=A1*y(:,i)+[b(1)*noise(1);b(2)*noise(2)];
end

%1st dimension MAX 
figure(1),subplot(2,2,1), histogram(x(1,:)), hold on, histogram(y(1,:)),histogram(xT(1,:)), legend("tULA","ULTAc","Exact")
subplot(2,1,2); plot(t,x(1,:),'r',t,xT(1,:),'g',t,y(1,:),'b:'), legend("tULA","ULTAc","Exact")
subplot(2,2,2), plot(x(1,:),x(2,:),'r',xT(1,:),xT(2,:),'g--',y(1,:),y(2,:),'b'),legend("tULA","ULTAc","Exact")

% 1st dimension
% figure(1),subplot(2,2,1), histogram(x(1,:)), hold on, histogram(y(1,:)),histogram(xc(1,:)), legend("tULA","tULAc","Exact")
% subplot(2,1,2); plot(t,x(1,:),'r',t,xc(1,:),'g',t,y(1,:),'b:'), legend("tULA","tULAc","Exact")
% subplot(2,2,2), plot(x(1,:),x(2,:),'r',xc(1,:),xc(2,:),'g--',y(1,:),y(2,:),'b'),legend("tULA","tULAc","Exact")

%2nd dimension
% figure(1),subplot(2,2,1), histogram(x(2,:)), hold on, histogram(y(2,:)),histogram(xc(2,:)), legend("tULA","tULAc","Exact")
% subplot(2,1,2); plot(t,x(2,:),'r',t,xc(2,:),'g',t,y(2,:),'b'), legend("tULA","tULAc","Exact")
% subplot(2,2,2), plot(x(1,:),x(2,:),'r',xc(1,:),xc(2,:),'g--',y(1,:),y(2,:),'b'),legend("tULA","tULAc","Exact")