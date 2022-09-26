---
title: "Stuff I Can Do on This Site"
date: "2021-11-17"
author: "Vien Vuong"
description: "A basic overview of syntax and formatting features this site supports. This article goes over basic Markdown and HTML, Hugo video shortcodes, placeholder text, KaTeX math typesetting, emoji support, Prism code formatting, and image rendering optimization."
tags: ["webdev"]
comments: false
socialShare: false
toc: false
math: true
---

A basic overview of syntax and formatting features this site supports. This article goes over basic Markdown and HTML, Hugo video shortcodes, placeholder text, KaTeX math typesetting, emoji support, Prism code formatting, and image rendering optimization.

# Markdown Syntax Guide

This article offers a sample of basic Markdown syntax that can be used in Hugo content files, also it shows whether basic HTML elements are decorated with CSS in a Hugo theme.

<!--more-->

## Headings

The following HTML `<h1>`—`<h6>` elements represent six levels of section headings. `<h1>` is the highest section level while `<h6>` is the lowest.

# H1

## H2

### H3

#### H4

##### H5

###### H6

## Paragraph

Xerum, quo qui aut unt expliquam qui dolut labo. Aque venitatiusda cum, voluptionse latur sitiae dolessi aut parist aut dollo enim qui voluptate ma dolestendit peritin re plis aut quas inctum laceat est volestemque commosa as cus endigna tectur, offic to cor sequas etum rerum idem sintibus eiur? Quianimin porecus evelectur, cum que nis nust voloribus ratem aut omnimi, sitatur? Quiatem. Nam, omnis sum am facea corem alique molestrunt et eos evelece arcillit ut aut eos eos nus, sin conecerem erum fuga. Ri oditatquam, ad quibus unda veliamenimin cusam et facea ipsamus es exerum sitate dolores editium rerore eost, temped molorro ratiae volorro te reribus dolorer sperchicium faceata tiustia prat.

Itatur? Quiatae cullecum rem ent aut odis in re eossequodi nonsequ idebis ne sapicia is sinveli squiatum, core et que aut hariosam ex eat.

## Blockquotes

The blockquote element represents content that is quoted from another source, optionally with a citation which must be within a `footer` or `cite` element, and optionally with in-line changes such as annotations and abbreviations.

#### Blockquote without attribution

> Tiam, ad mint andaepu dandae nostion secatur sequo quae.
> **Note** that you can use _Markdown syntax_ within a blockquote.

#### Blockquote with attribution

> Don't communicate by sharing memory, share memory by communicating.<br>
> — <cite>Rob Pike[^1]</cite>

[^1]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015.

## Tables

Tables aren't part of the core Markdown spec, but Hugo supports supports them out-of-the-box.

| Name  | Age |
| ----- | --- |
| Bob   | 27  |
| Alice | 23  |

#### Inline Markdown within tables

| Italics   | Bold     | Code   |
| --------- | -------- | ------ |
| _italics_ | **bold** | `code` |

## Code Blocks

#### Code block with backticks

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Example HTML5 Document</title>
  </head>
  <body>
    <p>Test</p>
  </body>
</html>
```

#### Code block indented with four spaces

    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Example HTML5 Document</title>
    </head>
    <body>
      <p>Test</p>
    </body>
    </html>

#### Code block with Hugo's internal highlight shortcode

{{< highlight html >}}

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Example HTML5 Document</title>
</head>
<body>
  <p>Test</p>
</body>
</html>
{{< /highlight >}}

## List Types

#### Ordered List

1. First item
2. Second item
3. Third item

#### Unordered List

- List item
- Another item
- And another item

#### Nested list

- Fruit
  - Apple
  - Orange
  - Banana
- Dairy
  - Milk
  - Cheese

## Other Elements — abbr, sub, sup, kbd, mark

<abbr title="Graphics Interchange Format">GIF</abbr> is a bitmap image format.

H<sub>2</sub>O

X<sup>n</sup> + Y<sup>n</sup> = Z<sup>n</sup>

Press <kbd><kbd>CTRL</kbd>+<kbd>ALT</kbd>+<kbd>Delete</kbd></kbd> to end the session.

Most <mark>salamanders</mark> are nocturnal, and hunt for insects, worms, and other small creatures.

---

---

# Emoji Support

Emoji can be enabled in a Hugo project in a number of ways.

<!--more-->

The [`emojify`](https://gohugo.io/functions/emojify/) function can be called directly in templates or [Inline Shortcodes](https://gohugo.io/templates/shortcode-templates/#inline-shortcodes).

To enable emoji globally, set `enableEmoji` to `true` in your site's [configuration](https://gohugo.io/getting-started/configuration/) and then you can type emoji shorthand codes directly in content files; e.g.

<p><span class="nowrap"><span class="emojify">🙈</span> <code>:see_no_evil:</code></span>  <span class="nowrap"><span class="emojify">🙉</span> <code>:hear_no_evil:</code></span>  <span class="nowrap"><span class="emojify">🙊</span> <code>:speak_no_evil:</code></span></p>
<br>

The [Emoji cheat sheet](http://www.emoji-cheat-sheet.com/) is a useful reference for emoji shorthand codes.

---

**N.B.** The above steps enable Unicode Standard emoji characters and sequences in Hugo, however the rendering of these glyphs depends on the browser and the platform. To style the emoji you can either use a third party emoji font or a font stack; e.g.

{{< highlight html >}}
.emoji {
font-family: Apple Color Emoji, Segoe UI Emoji, NotoColorEmoji, Segoe UI Symbol, Android Emoji, EmojiSymbols;
}
{{< /highlight >}}

{{< css.inline >}}

<style>
.emojify {
	font-family: Apple Color Emoji, Segoe UI Emoji, NotoColorEmoji, Segoe UI Symbol, Android Emoji, EmojiSymbols;
	font-size: 2rem;
	vertical-align: middle;
}
@media screen and (max-width:650px) {
  .nowrap {
    display: block;
    margin: 25px 0;
  }
}
</style>

{{< /css.inline >}}

---

---

# Prism Code Highlighting

This theme uses [Prism](https://prismjs.com/) for code highlighting. Other Hugo
themes usually include a pre-configured version of Prism, which complicates
updates and clutters the source code base with third-party JavaScript.

Only the Prism features you select in the Hugo site configuration are bundled by
the build process. This way, Prism can be easily updated with `npm` and the
size of the JavaScript and CSS bundles are minimized by only including what you
need.

<!--more-->

Here is a an example configuration demonstrating how to configure `languages`
and `plugins` in the `config.toml` file of your Hugo site:

```toml
[params]
  [params.prism]
    languages = [
      "markup",
      "css",
      "clike",
      "javascript"
    ]
    plugins = [
      "normalize-whitespace",
      "toolbar",
      "copy-to-clipboard"
    ]
```

## Languages

The following languages are available:

<!-- markdownlint-disable MD033 -->
<pre class="language-none" style="max-height: 500px">
  <code>
    {{% prism-features "languages" %}}
  </code>
</pre>
<!-- markdownlint-enable MD033 -->

## Plugins

Before using a plugin in production, read its documentation and test it
thoroughly. E.g., the [`remove-initial-line-feed` plugin](https://prismjs.com/plugins/remove-initial-line-feed/)
is still available despite being deprecated in favor of [`normalize-whitespace`](https://prismjs.com/plugins/normalize-whitespace/).

Many Prism plugins require using `<pre>` tags with custom attributes. Hugo uses
Goldmark as Markdown handler, which by default doesn't render raw inline HTML,
so make sure to enable [`unsafe`](https://gohugo.io/getting-started/configuration-markup#goldmark)
rendering if required:

```toml
[markup]
  [markup.goldmark]
    [markup.goldmark.renderer]
      unsafe = true
```

The following plugins are available:

```none
{{% prism-features "plugins" %}}
```

### Examples

#### Copy to Clipboard

`copy-to-clipboard` requires the `toolbar` plugin, so make sure to add it
**after** adding `toolbar` in the `config.toml` file:

Config:

```toml
[params.prism]
  # ...
  plugins = [
    "toolbar",
    "copy-to-clipboard"
  ]
```

#### Line Numbers

Config:

```toml
[params.prism]
  plugins = [
    "line-numbers"
  ]
```

Input:

```html
<pre class="line-numbers">
  <code>
    Example
  </code>
</pre>
```

Output:

<!-- markdownlint-disable MD033 -->
<pre class="line-numbers language-none" data-start="42">
  <code>
    Hello,
    World!

    Foo
    Bar
  </code>
</pre>
<!-- markdownlint-enable MD033 -->

#### Command Line

Config:

```toml
[params.prism]
  languages = [
    "bash"
  ]
  plugins = [
    "command-line"
  ]
```

Input:

```html
<pre class="command-line language-bash" data-user="root" data-host="localhost">
  <code>
    cd /usr/local/etc
    cp php.ini php.ini.bak
    vi php.ini
  </code>
</pre>

<pre
  class="command-line language-bash"
  data-user="chris"
  data-host="remotehost"
  data-output="2, 4-8"
>
  <code>
    pwd
    /usr/home/chris/bin
    ls -la
    total 2
    drwxr-xr-x   2 chris  chris     11 Jan 10 16:48 .
    drwxr--r-x  45 chris  chris     92 Feb 14 11:10 ..
    -rwxr-xr-x   1 chris  chris    444 Aug 25  2013 backup
    -rwxr-xr-x   1 chris  chris    642 Jan 17 14:42 deploy
  </code>
</pre>
```

Output:

<!-- markdownlint-disable MD033 -->
<pre class="command-line language-bash" data-user="root" data-host="localhost">
  <code>
    cd /usr/local/etc
    cp php.ini php.ini.bak
    vi php.ini
  </code>
</pre>

<pre
  class="command-line language-bash"
  data-user="chris"
  data-host="remotehost"
  data-output="2, 4-8"
>
  <code>
    pwd
    /usr/home/chris/bin
    ls -la
    total 2
    drwxr-xr-x   2 chris  chris     11 Jan 10 16:48 .
    drwxr--r-x  45 chris  chris     92 Feb 14 11:10 ..
    -rwxr-xr-x   1 chris  chris    444 Aug 25  2013 backup
    -rwxr-xr-x   1 chris  chris    642 Jan 17 14:42 deploy
  </code>
</pre>
<!-- markdownlint-enable MD033 -->

#### Diff Highlight

Config:

```toml
[params.prism]
  languages = [
    "javascript",
    "diff"
  ]
  plugins = [
    "diff-highlight"
  ]
```

Input:

```html
<pre class="language-diff-javascript diff-highlight">
  <code>
    @@ -4,6 +4,5 @@
    -    let foo = bar.baz([1, 2, 3]);
    -    foo = foo + 1;
    +    const foo = bar.baz([1, 2, 3]) + 1;
         console.log(`foo: ${foo}`);
  </code>
</pre>
```

Output:

<!-- markdownlint-disable MD033 -->
<pre class="language-diff-javascript diff-highlight">
  <code>
    @@ -4,6 +4,5 @@
    -    let foo = bar.baz([1, 2, 3]);
    -    foo = foo + 1;
    +    const foo = bar.baz([1, 2, 3]) + 1;
         console.log(`foo: ${foo}`);
  </code>
</pre>

## <!-- markdownlint-enable MD033 -->

---

---

# Math Typesetting

Mathematical notation in a Hugo project can be enabled by using third party JavaScript libraries.

<!--more-->

In this example we will be using [KaTeX](https://katex.org/)

- Create a partial under `/layouts/partials/math.html`
- Within this partial reference the [Auto-render Extension](https://katex.org/docs/autorender.html) or host these scripts locally.
- Include the partial in your templates like so:

```bash
{{ if or .Params.math .Site.Params.math }}
{{ partial "math.html" . }}
{{ end }}
```

- To enable KaTex globally set the parameter `math` to `true` in a project's configuration
- To enable KaTex on a per page basis include the parameter `math: true` in content files

**Note:** Use the online reference of [Supported TeX Functions](https://katex.org/docs/supported.html)

{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}

<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
{{ end }}
{{</ math.inline >}}

### Examples

{{< math.inline >}}

<p>
Inline math: \(\varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887…\)
</p>
{{</ math.inline >}}

Block math:

$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } }
$$

---

---

# Placeholder Text

Lorem est tota propiore conpellat pectoribus de pectora summo. <!--more-->Redit teque digerit hominumque toris verebor lumina non cervice subde tollit usus habet Arctonque, furores quas nec ferunt. Quoque montibus nunc caluere tempus inhospita parcite confusaque translucet patri vestro qui optatis lumine cognoscere flos nubis! Fronde ipsamque patulos Dryopen deorum.

1. Exierant elisi ambit vivere dedere
2. Duce pollice
3. Eris modo
4. Spargitque ferrea quos palude

Rursus nulli murmur; hastile inridet ut ab gravi sententia! Nomine potitus silentia flumen, sustinet placuit petis in dilapsa erat sunt. Atria tractus malis.

1. Comas hunc haec pietate fetum procerum dixit
2. Post torum vates letum Tiresia
3. Flumen querellas
4. Arcanaque montibus omnes
5. Quidem et

# Vagus elidunt

<svg class="canon" xmlns="http://www.w3.org/2000/svg" overflow="visible" viewBox="0 0 496 373" height="373" width="496"><g fill="none"><path stroke="#000" stroke-width=".75" d="M.599 372.348L495.263 1.206M.312.633l494.95 370.853M.312 372.633L247.643.92M248.502.92l246.76 370.566M330.828 123.869V1.134M330.396 1.134L165.104 124.515"></path><path stroke="#ED1C24" stroke-width=".75" d="M275.73 41.616h166.224v249.05H275.73zM54.478 41.616h166.225v249.052H54.478z"></path><path stroke="#000" stroke-width=".75" d="M.479.375h495v372h-495zM247.979.875v372"></path><ellipse cx="498.729" cy="177.625" rx=".75" ry="1.25"></ellipse><ellipse cx="247.229" cy="377.375" rx=".75" ry="1.25"></ellipse></g></svg>

[The Van de Graaf Canon](https://en.wikipedia.org/wiki/Canons_of_page_construction#Van_de_Graaf_canon)

## Mane refeci capiebant unda mulcebat

Victa caducifer, malo vulnere contra dicere aurato, ludit regale, voca! Retorsit colit est profanae esse virescere furit nec; iaculi matertera et visa est, viribus. Divesque creatis, tecta novat collumque vulnus est, parvas. **Faces illo pepulere** tempus adest. Tendit flamma, ab opes virum sustinet, sidus sequendo urbis.

Iubar proles corpore raptos vero auctor imperium; sed et huic: manus caeli Lelegas tu lux. Verbis obstitit intus oblectamina fixis linguisque ausus sperare Echionides cornuaque tenent clausit possit. Omnia putatur. Praeteritae refert ausus; ferebant e primus lora nutat, vici quae mea ipse. Et iter nil spectatae vulnus haerentia iuste et exercebat, sui et.

Eurytus Hector, materna ipsumque ut Politen, nec, nate, ignari, vernum cohaesit sequitur. Vel **mitis temploque** vocatus, inque alis, _oculos nomen_ non silvis corpore coniunx ne displicet illa. Crescunt non unus, vidit visa quantum inmiti flumina mortis facto sic: undique a alios vincula sunt iactata abdita! Suspenderat ego fuit tendit: luna, ante urbem Propoetides **parte**.

{{< css.inline >}}

<style>
.canon { background: white; width: 100%; height: auto; }
</style>

## {{< /css.inline >}}

---

---

# Rich Content

Hugo ships with several [Built-in Shortcodes](https://gohugo.io/content-management/shortcodes/#use-hugos-built-in-shortcodes) for rich content, along with a [Privacy Config](https://gohugo.io/about/hugo-and-gdpr/) and a set of Simple Shortcodes that enable static and no-JS versions of various social media embeds.

## <!--more-->

## YouTube Privacy Enhanced Shortcode

{{< youtube ZJthWmvUzzc >}}

<br>

---

## Twitter Simple Shortcode

{{< twitter_simple user="DesignReviewed" id="1085870671291310081" >}}

<br>

---

## Vimeo Simple Shortcode

{{< vimeo_simple 48912912 >}}

---

---

# Embed Video Files

Use the [video shortcode](https://github.com/schnerring/hugo-theme-gruvbox/blob/main/layouts/shortcodes/video.html)
to embed your video files from [Hugo Page Resources](https://gohugo.io/content-management/page-resources/).

{{< video src="my-video" autoplay="true" controls="false" loop="true" >}}

<!--more-->

With a page bundle looking like the following:

```text
feature-overview/
|-- index.md
|-- my-video.jpg
|-- my-video.mp4
|-- my-video.webm
```

You can embed `my-video` like this:

```markdown
{{</* video src="my-video" autoplay="true" controls="false" loop="true" */>}}
```

The shortcode looks for media files matching the filename `my-video*`. For each
`video` MIME type file, a `<source>` element is added. The first `image` MIME
type file is used as `poster` (thumbnail). It will render the following HTML:

```html
<video
  autoplay
  loop
  poster="/blog/feature-overview/assets/my-video.jpg"
  width="100%"
  playsinline
>
  <source src="/blog/feature-overview/assets/my-video.mp4" type="video/mp4" />
  <source src="/blog/feature-overview/assets/my-video.webm" type="video/webm" />
</video>
```

You can set a Markdown `caption`, wrapping the `<video>` inside a `<figure`>.

Additionally, the shortcode allows you to set the following attributes:

| Attribute   | Default |
| ----------- | ------- |
| autoplay    | `false` |
| controls    | `true`  |
| height      |         |
| loop        | `false` |
| muted       | `true`  |
| preload     |         |
| width       | `100%`  |
| playsinline | `true`  |

[Learn more about the `<video>` attributes here.](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/video#attributes)

---

---

# Image Optimization

The theme optimizes images by default with a custom [Hugo's markdown render hook](https://gohugo.io/getting-started/configuration-markup#markdown-render-hooks):

- The theme creates resized versions for each image, ranging from 100 to 700
  pixels wide.
- It generates [WebP](https://en.wikipedia.org/wiki/WebP) versions for each size
  if the original image format isn't WebP.
- The theme keeps the original file format as a fallback for browsers that
  [don't support the WebP format](https://caniuse.com/webp).
- Images in SVG format are embedded as-is.

## Blog Post Cover Images

Use the [front matter](https://gohugo.io/content-management/front-matter/) of
your posts to add cover images:

<!-- markdownlint-disable MD013 -->

```markdown
---
cover:
  src: assets/alexandre-van-thuan-mr9FouttLGY-unsplash.jpg
  alt: The interior of Stadsbiblioteket in Stockholm - Gunnar Asplunds library from 1928. The architecture is a transition between neoclassicism and functionalism.
  caption: By [Alexandre Van Thuan](https://unsplash.com/photos/mr9FouttLGY)
---
```

<!-- markdownlint-enable MD013 -->

## Captions

Add captions to your inline images like this:

```markdown
---
![Alt text](image-url.jpg "Caption with **markdown support**")
---
```

![The main library in Vancouver is architecturally significant. The angles and levels contour together to produce a trippy scene. It's pretty from the outside but stunning from the inside.](assets/alexandre-van-thuan-mr9FouttLGY-unsplash.jpg "The main library in Vancouver is architecturally significant. The angles and levels contour together to produce a trippy scene. It's pretty from the outside but stunning from the inside. By [Aaron Thomas](https://unsplash.com/photos/dMqlE7lgyOU)")

## JPEG and WebP Quality

The default quality is 75%. See the [official Image Processing Config Hugo docs](https://gohugo.io/content-management/image-processing/#image-processing-config).
Change it by adding the following to the `config.toml` file:

```toml
[imaging]
  quality = 75
```

## Resizing

By default, the theme creates resized versions of images ranging from 300 to 700
pixels wide in increments of 100 pixels. Override the resize behavior by
adding the following to the `config.toml` file:

```toml
[params]
  [params.imageResize]
    min = 300
    max = 700
    increment = 100
```

## Lazy Loading

Images are lazily loaded by default using the `loading="lazy"` attribute on
HTML `img` tags.

{{< video src="assets/lazy-loading" autoplay="true" controls="false" loop="true" >}}
