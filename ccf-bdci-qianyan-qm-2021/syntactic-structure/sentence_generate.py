import random

template = []


def get_template():
    global template

    sentences_together = [
        'X和Y能一起A吗',
        'X和YZ能一起A吗',
        'YZ和X能一起A吗',
        'YZ和X能一起AB吗',
        'Y和X能一起AB吗'
    ]
    sentences_trip = [
        'AB飞CD的EF',
        'AB飞CDEF',
        'AB到CD的EF',
        'AB到CDEF',
    ]
    sentences_compare = [
        'A比B更Z吗',
        'AB比C更Z',
        'A比BC更Z',
        'AB比CD更Z吗',
        'AB比CD更YZ吗',
        'A是B的C'
    ]
    sentences_active_passive = [
        'AB了C；C被AB了',
        'A把CB了；C被AB了',
        'A把CBD了；C被ABD了',
    ]

    template.append(sentences_together)
    template.append(sentences_trip)
    template.append(sentences_compare)
    template.append(sentences_active_passive)


if __name__ == '__main__':
    # get_template()
    # print(chr(65))
    template1 = ['5比2更3\t2比5更4\t1', '5比23\t2比54\t1', '5较2还3\t2较5更4\t1', '52较3还4\t3较52更4\t1']
    template2 = ['为什么是2\t为什么不2\t0', '为什么2\t为什么不2\t0', '为什么34\t为什么不34\t0', '为什么52不34\t52为什么不34\t1',
                 '为什么52不34\t52为什么不是34\t1', '为什么52要34\t52为什么34\t1',
                 '为什么52不34\t52为什么34\t0', '52为什么3\t52为什么不3\t0', '23是45的老公\t45的是23的老婆\t1', '23是43的老公\t45的是23的老婆\t0']
    template3 = ['5的6\t5以前的6\t0',
                 '5的6\t在5之后的6\t0',
                 '2020年的34\t2020年之前的34\t0',
                 '2020年的345\t2020年以前的345\t0',
                 '2021年之后的34\t2021年之前的34\t0',
                 '2021年的34\t2020年之后的34\t0',
                 '1980年的5\t1980年之后的34\t0'
                 ]
    template4 = ['2在34\t2刚刚在34\t0', '25在34\t25刚刚在34\t0', '2在34\t2刚刚34\t0', '2正34\t2刚才34\t0']
    template5 = ['怎么2不了\t这3怎么4不了\t0', '是2还是3请456\t46是2还是3\t1', '我想知道23还是45\t45还是23我想知道\t1']
    template6 = ['你知道25怎么3吗\t嗨 你知道25怎么3吗\t1', '打扰一下 23如何45\t请问23如何45\t1', '打扰一下 23如何45\t嗨 请问23如何45\t1',
                 '23 4如何5\t4如何5 23\t1', '怎么25不了\t34怎么25不了\t0', '怎么25\t34怎么25\t0', '2不了\t34怎么2不了\t0']
    template7 = ['23×45\t45×23\t1', '23×45\t54×23\t0', '34×5\t5×34\t1', '2.3×45\t3×2.45\t0', '45×2.3\t2.3×45\t1',
                 '45×23\t2.3×45\t0', '23×4.5\t2.3×45\t0', '23×4.5\t23×4.5\t1', '23×4.5\t24×3.5\t0',
                 '刘23ASDF\t刘43ASDF\t0', '李23asdf\t张23\t0', '王32asdf\t王35asdf\t0', '王34asdf\t王34asdf\t1',
                 '王5ASDF\t王5ASDF\t1', '李2ASDF\t张2ASDF\t0', '李2ASDF\t李2ASDF\t1', '2345年QWER\t2345QWER\t1',
                 '2345年QWER\t5342QWER\t0']
    template8 = ['23冷\t23冰凉\t1', '23热\t23冰凉\t0', '34凉爽\t34冰凉\t0', '防止52\t预防52\t1', '防止52\t避免52\t1',
                 '52和34一起6\t52和34同时6\t1', '5和3不同时6\t52和34一起6\t0', '52大学\t32大学\t0', 'xx7多少钱\txx72多少钱\t0',
                 '总是2345\t偶尔2345\t0', '经常52\t通常52\t1', '从不34\t总是34\t0', '经常52\t常常34\t0', '经常52\t常常52\t1']
    template9 = ['25可能34\t34可能25\t0', '234吧你这\t你这234吧\t1', '5423是不是？\t54是不是23？\t1', '他53是不是？\t你53是不是？\t0',
                 '5423是吗\t54是不是23\t1', '请2一234\t34请2一2\t1']
    result = []

    for i in range(1050):
        num1 = random.randint(0, 12)
        num2 = random.randint(13, 25)
        num3 = random.randint(0, 12)
        num4 = random.randint(13, 25)
        num5 = random.randint(0, 25)
        for j in range(len(template1)):
            temp = template1[j]
            num1 = random.randint(0, 12)
            num2 = random.randint(13, 25)
            num3 = random.randint(0, 12)
            num4 = random.randint(13, 25)
            temp = temp.replace('5', chr(65 + num1))
            temp = temp.replace('2', chr(65 + num2))
            temp = temp.replace('3', chr(97 + num3))
            temp = temp.replace('4', chr(97 + num4))
            result.append(temp)
            print(temp)
        for j in range(len(template2)):
            temp = template2[j]
            num1 = random.randint(0, 12)
            num2 = random.randint(13, 25)
            num3 = random.randint(0, 12)
            num4 = random.randint(13, 25)
            temp = temp.replace('5', chr(65 + num1))
            temp = temp.replace('2', chr(65 + num2))
            temp = temp.replace('3', chr(97 + num3))
            temp = temp.replace('4', chr(97 + num4))
            result.append(temp)
            print(temp)
        for j in range(len(template3)):
            temp = template3[j]
            num1 = random.randint(0, 12)
            num2 = random.randint(13, 25)
            num3 = random.randint(0, 12)
            num4 = random.randint(13, 25)
            temp = temp.replace('5', chr(65 + num1))
            temp = temp.replace('6', chr(65 + num2))
            temp = temp.replace('3', chr(97 + num3))
            temp = temp.replace('4', chr(97 + num4))
            result.append(temp)
            print(temp)
        for j in range(len(template4)):
            temp = template4[j]
            num1 = random.randint(0, 12)
            num2 = random.randint(13, 25)
            num3 = random.randint(0, 12)
            num4 = random.randint(13, 25)
            temp = temp.replace('5', chr(65 + num1))
            temp = temp.replace('2', chr(65 + num2))
            temp = temp.replace('3', chr(97 + num3))
            temp = temp.replace('4', chr(97 + num4))
            result.append(temp)
            print(temp)
        for j in range(len(template5)):
            temp = template5[j]
            num1 = random.randint(0, 12)
            num2 = random.randint(13, 25)
            num3 = random.randint(0, 12)
            num4 = random.randint(13, 25)
            temp = temp.replace('5', chr(65 + num1))
            temp = temp.replace('2', chr(65 + num2))
            temp = temp.replace('3', chr(97 + num3))
            temp = temp.replace('4', chr(97 + num4))
            temp = temp.replace('6', chr(65 + num4))
            result.append(temp)
            print(temp)
        for j in range(len(template6)):
            temp = template6[j]
            num1 = random.randint(0, 12)
            num2 = random.randint(13, 25)
            num3 = random.randint(0, 12)
            num4 = random.randint(13, 25)
            temp = temp.replace('5', chr(65 + num1))
            temp = temp.replace('2', chr(65 + num2))
            temp = temp.replace('3', chr(97 + num3))
            temp = temp.replace('4', chr(97 + num4))
            result.append(temp)
            print(temp)
        for j in range(len(template7)):
            temp = template7[j]
            num1 = random.randint(0, 12)
            num2 = random.randint(13, 25)
            num3 = random.randint(0, 12)
            num4 = random.randint(13, 25)
            temp = temp.replace('5', chr(65 + num1))
            temp = temp.replace('2', chr(65 + num2))
            temp = temp.replace('3', chr(97 + num3))
            temp = temp.replace('4', chr(97 + num4))
            result.append(temp)
            print(temp)
        for j in range(len(template8)):
            temp = template8[j]
            num1 = random.randint(0, 12)
            num2 = random.randint(13, 25)
            num3 = random.randint(0, 12)
            num4 = random.randint(13, 25)
            num5 = random.randint(0, 25)
            temp = temp.replace('5', chr(65 + num1))
            temp = temp.replace('2', chr(65 + num2))
            temp = temp.replace('3', chr(97 + num3))
            temp = temp.replace('4', chr(97 + num4))
            temp = temp.replace('6', chr(65 + num5))
            result.append(temp)
            print(temp)
        for j in range(len(template9)):
            temp = template9[j]
            num1 = random.randint(0, 12)
            num2 = random.randint(13, 25)
            num3 = random.randint(0, 12)
            num4 = random.randint(13, 25)
            temp = temp.replace('5', chr(65 + num1))
            temp = temp.replace('2', chr(65 + num2))
            temp = temp.replace('3', chr(97 + num3))
            temp = temp.replace('4', chr(97 + num4))
            result.append(temp)
            print(temp)

    with open(file='result/generate.csv', mode='w', encoding='utf8', newline='\n') as f:
        for row in result:
            f.write(row + '\n')
