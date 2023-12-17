解释：
xDarkLemon.github.io里面是源码，通过yarn run build命令生成静态页面，在dist文件夹中。
远端repo的master分支的内容就是生成的静态页面，即完全等同于dist文件夹。
之前用hexo，源码保存在source分支，hexo有个deploy的文件，push时不光将源码推送到远端source，并且将静态页面（在public中）推送到master分支。
现在基于vuepress的分支是vue-source。vuepress提供的deploy.sh不work，所以静态文件需要手动推送。在vue-source分支下生成dist文件夹后，把内容整个copy到xDarkLemon.github.io-master中（注意不要更改.git）。然后push到远端master即可。
也就是说，以前用hexo的时候，master和source分支都用这个本地的repo。现在用vuepress，源文件vue-source分支使用这个repo的，master分支使用xDarkLemon.github.io-master。因为需要手动copy文件，不好把他们放在同一个本地文件夹的两个分支中。

更改内容：
1. 修改docs中的内容 （https://vuepress-homepage.netlify.app/guide/customize.html#modification）
2. yarn dev 本地服务器看 localhost:8080
3. yarn run build 生成静态文件保存在dist中
4. 把xDarkLemon.github.io中vue-source的修改推送到远程vue-source分支
5. 把dist内容copy到xDarkLemon.github.io-master的master分支，把它推送到远程master分支

https://github.com/imfing/vuepress-homepage
https://vuepress-homepage.netlify.app/guide
https://vuepress.vuejs.org/guide/deploy.html#github-pages