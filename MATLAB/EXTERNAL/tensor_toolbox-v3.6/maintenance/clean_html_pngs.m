function clean_html_pngs(delfiles)
%CLEAN_HTML_PNGS Remove unused png image files from doc/html directory.
%
%   CLEAN_HTML_PNGS lists and removes any PNG file from ../doc/html that
%   is not referenced by a documentation files. This function must be run
%   from the tensor_toolbox/maintenance directory.
%
%   CLEAN_HTML_PNGS(false) lists but does not remove the files.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>


%%
htmldir = '../doc/html';
D = dir(htmldir);

%%
pngsinhtml = strings(0);
pngfiles = strings(0);
for i = 1:length(D)
    if length(D(i).name) < 4
        %fprintf('Other file: %s\n', D(i).name);
    elseif strcmp(D(i).name(end-3:end),'.png')
        %fprintf('PNG file: %s\n', D(i).name);
        pngfiles = cat(1,pngfiles,D(i).name);
    elseif strcmp(D(i).name(end-4:end),'.html')
        %fprintf('HTML file: %s\n', D(i).name);
        pngsinhtml = cat(1,pngsinhtml,list_html_pngs(D(i).name));
    else
        %fprintf('Other file: %s\n', D(i).name);
    end
end
%%
tf = ~ismember(pngfiles,pngsinhtml);
delpngfiles = pngfiles(tf);
fprintf('---Unneeded PNGs---\n')
for i = 1:length(delpngfiles)
    fprintf('%s\n', delpngfiles(i));
end

%%
if ~exist('delfiles','var') || delfiles == true
    for i = 1:length(delpngfiles)
        delete(fullfile(htmldir,delpngfiles(i)));
    end
end

