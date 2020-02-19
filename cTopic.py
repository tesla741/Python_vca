lt = ['check', 'disconnect', 'drop', 'cut-off', 'details', 'on hold', 'please hold',
      'please stay', 'hang on',
      'call you back',
      'get back',
      'be back',
      'reach you',
      'speak to',
      'call back',
      'put you on hold',
      'place your call on hold'
      ]


def cTopic(cmp_det, ch):
    ch_split = cmp_det.tokenize(ch)
    ch_str = " ".join(ch_split)
    c = 0
    for v in lt:
        if v in ch_str:
            c += 1
    if c >= 1:
        return True
    return False