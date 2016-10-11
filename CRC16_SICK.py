from ctypes import c_ushort

class CRC16_SICK(object):

    def updateCRC(self, crc, ch, prev_ch):
        short_c = 0x00ff & ch
        short_p = (0x00ff & prev_ch) << 8
        if crc & 0x8000:
            crc = (crc << 1) ^ 0x8005
        else:
            crc = crc << 1
        crc &= 0xffff
        crc ^= (short_c | short_p)
        return crc
                
                

    def calcSICK(self, string):
        crc = 0x0000
        prev_ch = 0
        for ch in string:
            crc = self.updateCRC(crc, ord(ch), prev_ch)
            prev_ch = ord(ch)
        crc1 = crc & 0x00ff
        crc2 = crc >> 8
        return [crc1, crc2]
