function s = circusjoin(strs, delim)
% Join multiple strings with a specified delimiter
%
%   s = circusjoin(strs, delim);
%       joins multiple strings in strs, which is a cell array, into 
%       a string, using delim as the delimiter.
%
%       If delim is omitted, it uses '' as delim, which means to 
%       simply concatenate all strings in strs.
%

%   History
%   -------
%       - Created by Dahua Lin, on Nov 10, 2010
%

%% verify input

if ~iscellstr(strs)
    error('circusjoin:invalidarg', 'strs should be a cell array of strings.');
end

if nargin < 2
    delim = '';
else
    if ~ischar(delim)
        error('circusjoin:invalidarg', 'delim should be a string.');
    end
end

%% main

n = numel(strs);

if n == 0
    s = '';
    
elseif n == 1
    s = strs{1};
    
else
    if isempty(delim)
        ss = strs;
    else
        ss = cell(1, 2*n-1);        
        ss(1:2:end) = strs(:);
        [ss{2:2:end}] = deal(delim);
    end
    
    s = [ss{:}];
end