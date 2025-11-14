long decode_c_version(long x, long y, long z)
{
    y -= z;
    x *= y;
    return x^(-( y & 1 ));

}
