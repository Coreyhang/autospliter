# n_layers, d_model, n_heads, d_head
def gpt_config_all():
    res = [gpt_small_config(),
           gpt_medium_config(),
           gpt_large_config(),
           gpt_xl_config(),
           gpt_2_7b(),
           gpt_6_7b(),
           gpt_13b(),
           gpt_175b()]
    return res


def gpt_small_config():
    return 'gpt_small', (12, 768, 12, 64)


def gpt_medium_config():
    return 'gpt_medium', (24, 1024, 16, 64)


def gpt_large_config():
    return 'gpt_large', (24, 1536, 16, 96)


def gpt_xl_config():
    return 'gpt_xl', (24, 2048, 16, 128)


def gpt_2_7b():
    return 'gpt_2_7b', (32, 2560, 32, 80)


def gpt_6_7b():
    return 'gpt_6_7b', (32, 4096, 32, 128)


def gpt_13b():
    return 'gpt_13b', (40, 5120, 40, 128)


def gpt_175b():
    return 'gpt_175b', (96, 12288, 96, 128)


if __name__ == '__main__':
    print(gpt_config_all())
