%%
function [X,Y]=GD_WY33(X,Y,a,b,a1,a2,a3,a4,b1,b2,b3,b4,u,rho)
f1=norm(a*X-b*Y)+1/(200*sum((a*X+b*Y).^2)-sum(a*X+b*Y)^2);
for k=1:1000
    %%%%%%%%%%%%%%%%%%%%%%
    tx0=a*X-b*Y;tx1=2-X-b1.*b1;tx2=X-b2.*b2;    tx3=a*X+b*Y;tx4=1/((tx3.^2)'*(200*ones(200,1))-sum(tx3))^2;
    delta_X=a2+a'*tx0/norm(tx0)-a1-rho*tx1/(2*norm(tx1))+rho*tx2/(2*norm(tx2))...
        -(tx4*a'*((200*ones(200,1).*tx3))-tx4*tx3'*(2*ones(200,1))*a'*ones(200,1));
    %%%
    X=X-u*delta_X;
    %%%%%%%%%%%%%%%%%%%%%%
    ty0=a*X-b*Y;ty1=2-Y-b3.*b3;ty2=Y-b4.*b4;    ty3=a*X+b*Y;ty4=1/((ty3.^2)'*(200*ones(200,1))-sum(ty3))^2;
    delta_Y=a4-b'*ty0/norm(ty0)-a3-rho*ty1/(2*norm(ty1))+rho*ty2/(2*norm(ty2))...
        -(ty4*b'*((200*ones(200,1).*ty3))-ty4*ty3'*(2*ones(200,1))*b'*ones(200,1));
    %%%
    Y=Y-u*delta_Y;
    %%%%%%%%%%%%%%%%%%%%%%
    delta_b1=-(2*a1.*b1+rho*(2-X-b1.*b1)/norm(2-X-b1.*b1));
    delta_b2=-(2*a2.*b2+rho*(X-b2.*b2)/norm(X-b2.*b2));
    delta_b3=-(2*a3.*b3+rho*(2-Y-b3.*b3)/norm(2-Y-b3.*b3));
    delta_b4=-(2*a4.*a4+rho*(Y-b4.*b4)/norm(Y-b4.*b4));
    %%%
    b1=b1-u*delta_b1;
    b2=b2-u*delta_b2;
    b3=b3-u*delta_b3;
    b4=b4-u*delta_b4;
    %%%%%%%%%%%%%%%%%%%%%%
    a1=max(a1+rho*(2-X-b1.^2),0);
    a2=max(a2+rho*(X-b2.^2),0);
    a3=max(a3+rho*(2-Y-b3.^2),0);
    a4=max(a4+rho*(Y-b4.^2),0);
    %%%%%%%%%%%%%%%
    X=X-u*delta_X;
    Y=Y-u*delta_Y;
    f2=norm(a*X-b*Y)+1/(200*sum((a*X+b*Y).^2)-sum(a*X+b*Y)^2);
    if norm(f1-f2)<0.0000001
        break;
    end
    f1=f2;
%     norm(f1-f2)
%     k
end
X=X;
Y=Y;
