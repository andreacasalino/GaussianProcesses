color 0b

del /s /f *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *.brf *.blg *.bbl *.lot *.lof

pdflatex Gaussian_Process.tex
bibtex   Gaussian_Process
pdflatex Gaussian_Process.tex
pdflatex Gaussian_Process.tex

del /s /f *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *.brf *.blg *.bbl *.lot *.lof
