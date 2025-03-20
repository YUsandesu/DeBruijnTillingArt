# 参考程序：

https://aatishb.com/patterncollider/

# 参考文献:

https://mathpages.com/home/kmath621/kmath621.htm

https://www.sciencedirect.com/science/article/pii/S0019357713000530
6.2. Inflation rules
**一种常用的创建大型镶嵌的方法是通过重复应用某些膨胀规则：将所有镶嵌片膨胀并用一组原始镶嵌片替换每个镶嵌片，遵循非常精确的规则。镶嵌百科全书给出了超过 180 个例子。**

# 备忘录:

https://www.math.utah.edu/~treiberg/PenroseSlides.pdf
本文28 开始讨论了PenroseTilling自相变换细分问题.

## 可以看的方向:

群论：分析 Penrose Tiling 的五次对称性，可用十次对称的二面体群研究。

除了关注的 P2 和 P3 类型，还有 P1, P4, P5 等不同类型的 Penrose tiling
**它们可以通过不同的 Cut and Project 方法或 Bruijn's grid 变体生成。**

充气和收缩 (Inflation and Deflation): 
Penrose tiling 具有自相似性，可以通过充气和收缩规则来生成和分析。研究 Bruijn's grid 方法如何体现充气和收缩规则是一个有趣的方向。

# 当前进度:
TODO _sort_girds_interaction 修改一下返回类型,使用pandas

https://www.adamponting.com/de-bruijn-tilings/
上文给出了一个全新的信息,**就算shift的距离不是相等的,创建的网络依然是可以平铺的**
**甚至可以是不完全等分gird**
我应该尝试思考为什么可以这样?

TODO 优化Gird创建,实际我们需要的只是点的序列而不是具体位置,而点的序列由于间隔一定的问题**一定是相同**的.
所以我们只需要求两条间隔的线在目标线的投影,就可以推导出点在目标线的序列

TODO 可以从任意一个交点开始,缩小1/pi倍,继续绘制网格,这样是可以继续细分的.
(奇数的origin_vector 会被自己的1/2细分 Penrose tiling另一个重要的自相似性特征——角度的递归细分)

TODO 之后Screen_draw应该加一个 内部变量 可以保存已经计算过的内容,这样不会每次在渲染之前重复计算

TODO 调整gird 返回的 数据类型

# 效果参考:
https://www.instagram.com/p/CFe4yYBnuHS/
![节点风格.png](picture%2F%E8%8A%82%E7%82%B9%E9%A3%8E%E6%A0%BC.png)

# 附:

"Symmetry of Tilings of the Plane" by Branko Grünbaum and G. C. Shephard: 这本书的章节中关于 Penrose tiling 的部分，会详细介绍 kite 和 dart 的 deflation 规则，以及角度在 deflation 过程中的变化。

"Penrose Tiles to Trapdoor Ciphers" by Martin Gardner (Scientific American, January 1977): 虽然是科普文章，但 Gardner 的文章是 Penrose tiling 早期非常重要的介绍，其中用清晰的图示解释了 kite 和 dart 的 deflation 过程。

"The emperor's new mind: Concerning computers, minds, and the laws of physics" by Roger Penrose: Penrose 在这本书中也详细解释了 Penrose tiling，包括 kite 和 dart 以及 deflation 过程。

"The Golden Ratio: The Story of Phi, the World's Most Astonishing Number" by Mario Livio: 这本书深入探讨了黄金分割比的数学和历史，包括它在 Penrose tiling 和黄金三角形中的应用。
"Geometry and the Visual Arts" by Daniel Pedoe: 这本书探讨了几何学在艺术中的应用，包括黄金分割比和 Penrose tiling。
"Penrose Tiles Talk" by Martin Gardner: 这篇文章以通俗易懂的方式介绍了 Penrose tiling 和黄金分割比。

# 想要读的书(暂时看不到)：

《Tilings and Patterns》（Branko Grünbaum 和 G.C. Shephard 著），详细介绍铺砖理论。
https://link.springer.com/chapter/10.1007/978-3-031-28428-1_5
本章介绍了 Nicolaas G. de Bruijn 引入的方法和理念，他为 Penrose 镶嵌的研究提供了一些主要贡献。
在这里，我们学习如何将一个正五格网转化为一个 Penrose 镶嵌图案。
我们讨论了切割-投影法，首先是针对一维镶嵌，然后是 Penrose 镶嵌。
我们学习如何通过将二维晶格的一部分投影到一条线上来构建斐波那契镶嵌图案，
以及如何通过将五维晶格的一部分投影到一个平面上来构建由 Penrose 菱形构成的镶嵌图案。
我们研究了一个五格网在组合过程中如何变化，并分类与正五格网相关的镶嵌。

Kurt Bruckner’s view on the Penrose tiling
https://link.springer.com/article/10.1007/s11224-016-0790-1?fromPaywallRec=true