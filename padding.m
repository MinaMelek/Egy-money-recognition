function [ K_pad ] = padding( imag )
s = size(imag);
m=s(1);n=s(2);
p = 292;
q = 548;
% K_pad = padarray(K, [(p-m)/2 (q-n)/2], 'replicate');
K_pad = padarray(imag, [floor((p-m)/2) floor((q-n)/2)],'post');
K_pad = padarray(K_pad, [ceil((p-m)/2) ceil((q-n)/2)],'pre');
end