def remove_labels(article):
    pf_label_keywords = ['true', 'false', 'pants']
    tmp_split = article.split('.')
    tmp_article_without_label = ''
    for i in range(len(tmp_split)):
        for keyword in pf_label_keywords:
            if keyword in tmp_split[-i - 1].lower():
                tmp_article_without_label = '.'.join(tmp_split[:-i - 1])
                tmp_article_without_label += '.'
                return tmp_article_without_label