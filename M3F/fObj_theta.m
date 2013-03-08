function f = fObj_theta(theta, i, r, Y, Su, U, V, h)
T_j = Y(i,Su{i}) <= r;
UV = U(i,:)*V(Su{i},:)';
f = sum(max(0,h+UV(T_j)-theta)) + sum(max(h-UV(~T_j)+theta,0));
end
