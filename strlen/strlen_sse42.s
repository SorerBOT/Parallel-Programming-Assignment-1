.section .data


.section .text
.globl strlen_sse42
.type strlen_sse42, @function

strlen_sse42:
    pushq       %rcx

    xorq        %rcx, %rcx
    pxor        %xmm0, %xmm0            # null terminator string
    xorq        %rdx, %rdx              # %rdx will be used in order to loop over the string
search_segment:                         # searches segments of 16 bytes trying to find the first occurrence of a null terminator
    movdqu      (%rdi, %rdx), %xmm1     # moving the first 16 bytes of the string into %xmm1
    xorq        %rcx, %rcx              # %rcx would contain the index the first null terminator occurrence should it be found
    pcmpistri   $0x08, %xmm0, %xmm1     # try finding an occurrence of null terminator, if found, the index of the occurrence will be inserted into %eax
    addq        %rcx, %rdx              # adding adding the index to the total length (or 16 if not found)
    cmpq        $16, %rcx               # checking if a null terminator was found or not
    je          search_segment
    jmp found_match

found_match:
    movq        %rdx, %rax       

    popq        %rcx
    ret
