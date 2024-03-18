function pngs = list_html_pngs(fname)
%LIST_HTML_PNGS List png image files in documentation page.
%
%   LIST_HTML_PNGS(FNAME) lists the png image files references in FNAME,
%   which should be an HTML file in ../doc/html. This function must be run
%   from the tensor_toolbox/maintenance directory.
%
%   See also CLEAN_HTML_PNGS.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

% Hard-coded directory name
dirname = '../doc/html';

% Read the HTML file
html_file = fullfile(dirname,fname);
html_text = fileread(html_file);

% Extract all image file names using regular expressions
img_pattern = '<img[^>]*src="([^"]+)"[^>]*>';
img_matches = regexp(html_text, img_pattern, 'tokens');
pngs = strings(numel(img_matches),1);

% Display the image file names
for i = 1:numel(img_matches)
    img_src = img_matches{i}{1};
    [~, img_name, img_ext] = fileparts(img_src);

    img_file = [img_name, img_ext];
    pngs(i,1) = img_file;
end