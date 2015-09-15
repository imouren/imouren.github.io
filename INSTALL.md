
# how to create this project?

## init project
```
echo # imouren.github.io >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/imouren/imouren.github.io.git
git push -u origin master
```
## add html page
```
touch index.html
echo "Hello World" > index.html
git push -u origin master
```
## show this page

brew on http://username.github.io/index.html

## bound a custom domain

```
touch CNAME
echo yourdomain > CNAME
git push -u origin master
configue a CNAME yourdomian -> username.github.io
```

