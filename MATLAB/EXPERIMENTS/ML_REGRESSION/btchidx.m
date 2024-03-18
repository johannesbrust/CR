function [si, ei] = btchidx(i, btch, N)
%BTCHIDX Computes start and end indices to access data
%   [si, ei] = btchidx(i, btch, N) computes the start and end index of 
%   batches of data pairs, based on the index of batch i, its size
%   btch and the total number of data pairs.
% 
%--------------------------------------------------------------------------
% 01/20/24, J.B., initial implementation

si = (i-1)*btch + 1;
ei = si + btch;

if ei > N; ei = N; end

end