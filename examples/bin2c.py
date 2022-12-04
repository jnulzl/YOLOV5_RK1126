import os
import sys
import argparse


def convert_func(src_path, des_path, upper, align_byte = 16, step = 12):
    array_name = os.path.basename(src_path).replace('.', '_')
    count = 1
    with open(des_path, 'wb') as result_file:
        result_file.write(b'#ifndef __%s__H__\n' % array_name.upper().encode('utf-8'))
        result_file.write(b'#define __%s__H__\n' % array_name.upper().encode('utf-8'))
        result_file.write(b'\n')

        result_file.write(b'#ifdef __cplusplus\n')
        result_file.write(b'extern "C" {\n')
        result_file.write(b'#endif\n')
        result_file.write(b'\n')

        result_file.write(b'#if defined(_MSC_VER)\n')
        result_file.write(b'    #define BIN2ARRAY_ALIGN __declspec(align(%d))\n' % align_byte)
        result_file.write(b'#else\n')
        result_file.write(b'    #define BIN2ARRAY_ALIGN  __attribute__((__aligned__(%d)))\n' % align_byte)
        result_file.write(b'#endif\n\n')
        result_file.write(b'const static unsigned char %s[] = {\n  ' % array_name.encode('utf-8'))

        buffer_data = open(src_path, 'rb').read()
        last_byte = buffer_data[-1]
        buffer_data = buffer_data[:-1]
        if upper:
            for index, b in enumerate(buffer_data):
                result_file.write(b'0x%02X, ' % b)
                if 0 == count % step:
                    result_file.write(b'\n')
                    result_file.write(b'  ')
                count += 1   
            result_file.write(b'0x%02X\n' % last_byte)
        else:
            for index, b in enumerate(buffer_data):
                result_file.write(b'0x%02x, ' % b)
                if 0 == count % step:
                    result_file.write(b'\n')
                    result_file.write(b'  ')
                count += 1   
            result_file.write(b'0x%02x\n' % last_byte)


        result_file.write(b'};\n')
        result_file.write(b'const static unsigned int %s_len = %d;\n' % (array_name.encode('utf-8'), count))
        result_file.write(b'\n')

        result_file.write(b'#ifdef __cplusplus\n')
        result_file.write(b'}\n')
        result_file.write(b'#endif // __cplusplus\n')

        result_file.write(b'\n')
        result_file.write(b'#endif // %s\n' % array_name.upper().encode('utf-8'))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert general file to C language include file')
    # general
    parser.add_argument('--src_path', 
                         default='', 
                         type = str,
                         help='The source file path.')
    parser.add_argument('--des_path',
                        default='',
                        type = str,
                        help='The des file path.')
    parser.add_argument('--upper', 
                         action='store_true', 
                         help='use upper case hex letters.')
    parser.add_argument('--num_line', 
                         type=int, 
                        default=12,
                         help='The number of hex letters each line.')
    parser.add_argument('--align_byte', 
                         type=int, 
                        default=16,
                         help='The align byte number.')

    opt = parser.parse_args()

    if not os.path.exists(opt.src_path):
        print("Please see: python %s --help" % sys.argv[0])
        sys.exit(0)

    opt.des_path = opt.src_path.replace('.', '_') + '.h' if '' == opt.des_path else opt.des_path
    convert_func(opt.src_path, opt.des_path, opt.upper, align_byte = opt.align_byte, step = opt.num_line)
