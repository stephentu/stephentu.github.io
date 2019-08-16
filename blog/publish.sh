#!/bin/bash

jekyll build

# fix bug in jekyll-feed not respecting baseurl
sed -i.bak 's/blog\/blog/blog/g' _site/rss.xml
rm _site/rss.xml.bak

rm -rf ~/stephentu.github.io/blog/*
cp -R _site/* ~/stephentu.github.io/blog
