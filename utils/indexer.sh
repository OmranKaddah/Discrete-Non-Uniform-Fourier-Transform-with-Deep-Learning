cd dataset
cd images
cd train
ls -v | cat -n | while read n f; do mv -n "$f" "$n.jpg"; done 
cd ..
cd test
ls -v | cat -n | while read n f; do mv -n "$f" "$n.jpg"; done 
cd ..
cd ..

