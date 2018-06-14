# Helper functions to transform between text<->labels
# Source: https://github.com/rizkiarm/LipNet/blob/master/lipnet/lipreading/helpers.py

def text_to_labels(text: str) -> [chr]:
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret


def labels_to_text(labels: [chr]) -> str:
    # 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        if c >= 0 and c < 26:
            text += chr(c + ord('a'))
        elif c == 26:
            text += ' '
    return text


if __name__ == '__main__':
    text = 'hello sailor!'

    labels = text_to_labels(text)
    print('labels: {}'.format(labels))

    text_back = labels_to_text(labels)
    print('text_back: {}'.format(text_back))
