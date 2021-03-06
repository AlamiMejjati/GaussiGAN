import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br, video
import os
import skvideo.io


class HTML:
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        # self.img_dir = os.path.join(self.web_dir, 'images')
        # if not os.path.exists(self.web_dir):
        #     os.makedirs(self.web_dir)
        # if not os.path.exists(self.img_dir):
        #     os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.web_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")

        self.doc.add(self.t)

    def add_text(self, txts, width=400):
        self.add_table()
        with self.t:
            with tr():
                for txt in txts:
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p(style="width:%dpx" % width):
                            p(h3(txt))

    def add_text_sep(self, txts, width=400):
        self.add_table()
        with self.t:
            with tr():
                for txt in txts:
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p(style="width:%dpx" % width):
                            splitted = txt.split(',')
                            if len(splitted) > 1:
                                for h in splitted:
                                    p(h3(h))
                            else:
                                p(h3(txt))

                            # p(h3(txt))

    def add_images(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=link):
                                img(style="width:%dpx" % width, src=link)
                            br()
                            p(h3(txt))
    def add_im_vid(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    if im[-3:] != 'mp4':
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                with a(href=link):
                                    img(style="width:%dpx" % width, src=link)
                                br()
                                splitted = txt.split(' or ')
                                if len(splitted) > 1:
                                    for h in splitted:
                                        p(h3(h))
                                else:
                                    p(h3(txt))

                    else:
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                with a(href=link):
                                    video(style="width:%dpx" % width, src=link, loop=True, autoplay=True, controls=True, muted=True)
                                br()
                                p(h3(txt))

    def add_images_filterchart(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=link):
                                img(style="width:%dpx" % width, src=link)
                            br()
                            splitted = txt.split('separator')
                            if len(splitted) > 1:
                                for h in splitted:
                                    p(h)
                            else:
                                p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
