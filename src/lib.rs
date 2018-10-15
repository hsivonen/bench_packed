// Copyright 2015-2016 Mozilla Foundation. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(platform_intrinsics, test)]

extern crate simd;

#[macro_use(shuffle)]
extern crate packed_simd;

extern crate test;

const MEM_INNER_LOOP: u64 = 500;

pub const SIMD_STRIDE_SIZE: usize = 16;

pub const SIMD_ALIGNMENT_MASK: usize = 15;

const SIMD_ALIGNMENT: usize = 16;

fn round_str_index_up(s: &str, i: usize) -> usize {
    let b = s.as_bytes();
    let mut idx = i;
    while (idx < b.len()) && ((b[idx] & 0xC0) == 0x80) {
        idx += 1;
    }
    idx
}

mod old {
    use super::*;

    use self::test::Bencher;

    use simd::u16x8;
    use simd::u8x16;
    use simd::i8x16;
    use simd::Simd;

    extern "platform-intrinsic" {
        fn simd_shuffle16<T: Simd, U: Simd<Elem = T::Elem>>(x: T, y: T, idx: [u32; 16]) -> U;
    }

    extern "platform-intrinsic" {
        fn x86_mm_movemask_epi8(x: i8x16) -> i32;
    }

    pub fn simd_is_ascii(s: u8x16) -> bool {
        unsafe {
            let signed: i8x16 = ::std::mem::transmute_copy(&s);
            x86_mm_movemask_epi8(signed) == 0
        }
    }

    #[inline(always)]
    pub fn simd_is_str_latin1(s: u8x16) -> bool {
        if simd_is_ascii(s) {
            return true;
        }
        let above_str_latin1 = u8x16::splat(0xC4);
        s.lt(above_str_latin1).all()
    }

    #[inline(always)]
    pub fn simd_unpack(s: u8x16) -> (u16x8, u16x8) {
        unsafe {
            let first: u8x16 = simd_shuffle16(
                s,
                u8x16::splat(0),
                [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23],
            );
            let second: u8x16 = simd_shuffle16(
                s,
                u8x16::splat(0),
                [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31],
            );
            (
                ::std::mem::transmute_copy(&first),
                ::std::mem::transmute_copy(&second),
            )
        }
    }

    // Below this line no difference between old and packed modules

    #[inline(never)]
    pub fn is_str_latin1(buffer: &str) -> bool {
        is_str_latin1_impl(buffer).is_none()
    }

    #[inline(always)]
    fn is_str_latin1_impl(buffer: &str) -> Option<usize> {
        let mut offset = 0usize;
        let bytes = buffer.as_bytes();
        let len = bytes.len();
        if len >= SIMD_STRIDE_SIZE {
            let src = bytes.as_ptr();
            let mut until_alignment = (SIMD_ALIGNMENT - ((src as usize) & SIMD_ALIGNMENT_MASK)) &
                                       SIMD_ALIGNMENT_MASK;
            if until_alignment + SIMD_STRIDE_SIZE <= len {
                while until_alignment != 0 {
                    if bytes[offset] > 0xC3 {
                        return Some(offset);
                    }
                    offset += 1;
                    until_alignment -= 1;
                }
                let len_minus_stride = len - SIMD_STRIDE_SIZE;
                loop {
                    if !simd_is_str_latin1(unsafe { *(src.offset(offset as isize) as *const u8x16) }) {
                        // TODO: Ensure this compiles away when inlined into `is_str_latin1()`.
                        while bytes[offset] & 0xC0 == 0x80 {
                            offset += 1;
                        }
                        return Some(offset);
                    }
                    offset += SIMD_STRIDE_SIZE;
                    if offset > len_minus_stride {
                        break;
                    }
                }
            }
        }
        for i in offset..len {
            if bytes[i] > 0xC3 {
                return Some(i);
            }
        }
        None
    }

    #[bench]
    fn bench_mem_is_str_latin1_true_1000(b: &mut Bencher) {
        let s = include_str!("wikipedia/de-edit.txt");
        let mut string = String::with_capacity(s.len());
        string.push_str(s);
        let truncated = &string[..round_str_index_up(&string[..], 1000)];
        b.bytes = (truncated.len() as u64) * MEM_INNER_LOOP;
        b.iter(|| {
            for _ in 0..MEM_INNER_LOOP {
                test::black_box(is_str_latin1(test::black_box(truncated)));
            }
        });
    }

    #[inline(always)]
    pub unsafe fn load16_aligned(ptr: *const u8) -> u8x16 {
        *(ptr as *const u8x16)
    }

    #[inline(always)]
    pub unsafe fn store8_unaligned(ptr: *mut u16, s: u16x8) {
        ::std::ptr::copy_nonoverlapping(&s as *const u16x8 as *const u8, ptr as *mut u8, 16);
    }

    #[inline(always)]
    pub unsafe fn store8_aligned(ptr: *mut u16, s: u16x8) {
        *(ptr as *mut u16x8) = s;
    }

    #[inline(always)]
    pub unsafe fn unpack_stride_both_aligned(src: *const u8, dst: *mut u16) {
        let simd = load16_aligned(src);
        let (first, second) = simd_unpack(simd);
        store8_aligned(dst, first);
        store8_aligned(dst.offset(8), second);
    }

    #[inline(always)]
    pub unsafe fn unpack_stride_src_aligned(src: *const u8, dst: *mut u16) {
        let simd = load16_aligned(src);
        let (first, second) = simd_unpack(simd);
        store8_unaligned(dst, first);
        store8_unaligned(dst.offset(8), second);
    }

    #[inline(always)]
    pub unsafe fn unpack_latin1(src: *const u8, dst: *mut u16, len: usize) {
        let unit_size = ::std::mem::size_of::<u8>();
        let mut offset = 0usize;
        if SIMD_STRIDE_SIZE <= len {
            let mut until_alignment = ((SIMD_STRIDE_SIZE
                - ((src as usize) & SIMD_ALIGNMENT_MASK))
                & SIMD_ALIGNMENT_MASK)
                / unit_size;
            while until_alignment != 0 {
                *(dst.offset(offset as isize)) = *(src.offset(offset as isize)) as u16;
                offset += 1;
                until_alignment -= 1;
            }
            let len_minus_stride = len - SIMD_STRIDE_SIZE;
            if offset + SIMD_STRIDE_SIZE * 2 <= len {
                let len_minus_stride_times_two = len_minus_stride - SIMD_STRIDE_SIZE;
                if (dst.offset(offset as isize) as usize) & SIMD_ALIGNMENT_MASK == 0 {
                    loop {
                        unpack_stride_both_aligned(
                            src.offset(offset as isize),
                            dst.offset(offset as isize),
                        );
                        offset += SIMD_STRIDE_SIZE;
                        unpack_stride_both_aligned(
                            src.offset(offset as isize),
                            dst.offset(offset as isize),
                        );
                        offset += SIMD_STRIDE_SIZE;
                        if offset > len_minus_stride_times_two {
                            break;
                        }
                    }
                } else {
                    loop {
                        unpack_stride_src_aligned(
                            src.offset(offset as isize),
                            dst.offset(offset as isize),
                        );
                        offset += SIMD_STRIDE_SIZE;
                        unpack_stride_src_aligned(
                            src.offset(offset as isize),
                            dst.offset(offset as isize),
                        );
                        offset += SIMD_STRIDE_SIZE;
                        if offset > len_minus_stride_times_two {
                            break;
                        }
                    }
                }
            }
            if offset < len_minus_stride {
                unpack_stride_src_aligned(src.offset(offset as isize), dst.offset(offset as isize));
                offset += SIMD_STRIDE_SIZE;
            }
        }
        while offset < len {
            let code_unit = *(src.offset(offset as isize));
            // On x86_64, this loop autovectorizes but in the pack
            // case there are instructions whose purpose is to make sure
            // each u16 in the vector is truncated before packing. However,
            // since we don't care about saturating behavior of SSE2 packing
            // when the input isn't Latin1, those instructions are useless.
            // Unfortunately, using the `assume` intrinsic to lie to the
            // optimizer doesn't make LLVM omit the trunctation that we
            // don't need. Possibly this loop could be manually optimized
            // to do the sort of thing that LLVM does but without the
            // ANDing the read vectors of u16 with a constant that discards
            // the high half of each u16. As far as I can tell, the
            // optimization assumes that doing a SIMD read past the end of
            // the array is OK.
            *(dst.offset(offset as isize)) = code_unit as u16;
            offset += 1;
        }
    }

    #[inline(never)]
    pub fn convert_latin1_to_utf16(src: &[u8], dst: &mut [u16]) {
        assert!(
            dst.len() >= src.len(),
            "Destination must not be shorter than the source."
        );
        // TODO: On aarch64, the safe version autovectorizes to the same unpacking
        // instructions and this code, but, yet, the autovectorized version is
        // faster.
        unsafe {
            unpack_latin1(src.as_ptr(), dst.as_mut_ptr(), src.len());
        }
    }

    #[bench]
    fn bench_mem_convert_latin1_to_utf16_1000(b: &mut Bencher) {
        let bytes = include_bytes!("wikipedia/de-edit.txt");
        let mut v = Vec::with_capacity(bytes.len());
        v.extend_from_slice(bytes);
        let truncated = &v[..1000];
        let capacity = 1000 * 4;
        let mut vec = Vec::with_capacity(capacity);
        vec.resize(capacity, 0u16);
        let dst = &mut vec[..];
        b.bytes = (truncated.len() as u64) * MEM_INNER_LOOP;
        b.iter(|| {
            for _ in 0..MEM_INNER_LOOP {
                test::black_box(convert_latin1_to_utf16(test::black_box(truncated), test::black_box(dst)));
            }
        });
    }

}

mod packed {
    use super::*;

    use self::test::Bencher;

    use packed_simd::u16x8;
    use packed_simd::u8x16;
    use packed_simd::FromBits;

    use std::arch::x86_64::__m128i;
    use std::arch::x86_64::_mm_movemask_epi8;

    #[inline(always)]
    pub fn simd_is_ascii(s: u8x16) -> bool {
        unsafe {
            _mm_movemask_epi8(__m128i::from_bits(s)) == 0
        }
    }

    #[inline(always)]
    pub fn simd_is_str_latin1(s: u8x16) -> bool {
        if simd_is_ascii(s) {
            return true;
        }
        let above_str_latin1 = u8x16::splat(0xC4);
        s.lt(above_str_latin1).all()
    }

    #[inline(always)]
    pub fn simd_unpack(s: u8x16) -> (u16x8, u16x8) {
        unsafe {
            let first: u8x16 = shuffle!(
                s,
                u8x16::splat(0),
                [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
            );
            let second: u8x16 = shuffle!(
                s,
                u8x16::splat(0),
                [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
            );
            (
                ::std::mem::transmute_copy(&first),
                ::std::mem::transmute_copy(&second),
            )
        }
    }

    // Below this line no difference between old and packed modules

    #[inline(never)]
    pub fn is_str_latin1(buffer: &str) -> bool {
        is_str_latin1_impl(buffer).is_none()
    }

    #[inline(always)]
    fn is_str_latin1_impl(buffer: &str) -> Option<usize> {
        let mut offset = 0usize;
        let bytes = buffer.as_bytes();
        let len = bytes.len();
        if len >= SIMD_STRIDE_SIZE {
            let src = bytes.as_ptr();
            let mut until_alignment = (SIMD_ALIGNMENT - ((src as usize) & SIMD_ALIGNMENT_MASK)) &
                                       SIMD_ALIGNMENT_MASK;
            if until_alignment + SIMD_STRIDE_SIZE <= len {
                while until_alignment != 0 {
                    if bytes[offset] > 0xC3 {
                        return Some(offset);
                    }
                    offset += 1;
                    until_alignment -= 1;
                }
                let len_minus_stride = len - SIMD_STRIDE_SIZE;
                loop {
                    if !simd_is_str_latin1(unsafe { *(src.offset(offset as isize) as *const u8x16) }) {
                        // TODO: Ensure this compiles away when inlined into `is_str_latin1()`.
                        while bytes[offset] & 0xC0 == 0x80 {
                            offset += 1;
                        }
                        return Some(offset);
                    }
                    offset += SIMD_STRIDE_SIZE;
                    if offset > len_minus_stride {
                        break;
                    }
                }
            }
        }
        for i in offset..len {
            if bytes[i] > 0xC3 {
                return Some(i);
            }
        }
        None
    }

    #[bench]
    fn bench_mem_is_str_latin1_true_1000(b: &mut Bencher) {
        let s = include_str!("wikipedia/de-edit.txt");
        let mut string = String::with_capacity(s.len());
        string.push_str(s);
        let truncated = &string[..round_str_index_up(&string[..], 1000)];
        b.bytes = (truncated.len() as u64) * MEM_INNER_LOOP;
        b.iter(|| {
            for _ in 0..MEM_INNER_LOOP {
                test::black_box(is_str_latin1(test::black_box(truncated)));
            }
        });
    }

    #[inline(always)]
    pub unsafe fn load16_aligned(ptr: *const u8) -> u8x16 {
        *(ptr as *const u8x16)
    }

    #[inline(always)]
    pub unsafe fn store8_unaligned(ptr: *mut u16, s: u16x8) {
        ::std::ptr::copy_nonoverlapping(&s as *const u16x8 as *const u8, ptr as *mut u8, 16);
    }

    #[inline(always)]
    pub unsafe fn store8_aligned(ptr: *mut u16, s: u16x8) {
        *(ptr as *mut u16x8) = s;
    }

    #[inline(always)]
    pub unsafe fn unpack_stride_both_aligned(src: *const u8, dst: *mut u16) {
        let simd = load16_aligned(src);
        let (first, second) = simd_unpack(simd);
        store8_aligned(dst, first);
        store8_aligned(dst.offset(8), second);
    }

    #[inline(always)]
    pub unsafe fn unpack_stride_src_aligned(src: *const u8, dst: *mut u16) {
        let simd = load16_aligned(src);
        let (first, second) = simd_unpack(simd);
        store8_unaligned(dst, first);
        store8_unaligned(dst.offset(8), second);
    }

    #[inline(always)]
    pub unsafe fn unpack_latin1(src: *const u8, dst: *mut u16, len: usize) {
        let unit_size = ::std::mem::size_of::<u8>();
        let mut offset = 0usize;
        if SIMD_STRIDE_SIZE <= len {
            let mut until_alignment = ((SIMD_STRIDE_SIZE
                - ((src as usize) & SIMD_ALIGNMENT_MASK))
                & SIMD_ALIGNMENT_MASK)
                / unit_size;
            while until_alignment != 0 {
                *(dst.offset(offset as isize)) = *(src.offset(offset as isize)) as u16;
                offset += 1;
                until_alignment -= 1;
            }
            let len_minus_stride = len - SIMD_STRIDE_SIZE;
            if offset + SIMD_STRIDE_SIZE * 2 <= len {
                let len_minus_stride_times_two = len_minus_stride - SIMD_STRIDE_SIZE;
                if (dst.offset(offset as isize) as usize) & SIMD_ALIGNMENT_MASK == 0 {
                    loop {
                        unpack_stride_both_aligned(
                            src.offset(offset as isize),
                            dst.offset(offset as isize),
                        );
                        offset += SIMD_STRIDE_SIZE;
                        unpack_stride_both_aligned(
                            src.offset(offset as isize),
                            dst.offset(offset as isize),
                        );
                        offset += SIMD_STRIDE_SIZE;
                        if offset > len_minus_stride_times_two {
                            break;
                        }
                    }
                } else {
                    loop {
                        unpack_stride_src_aligned(
                            src.offset(offset as isize),
                            dst.offset(offset as isize),
                        );
                        offset += SIMD_STRIDE_SIZE;
                        unpack_stride_src_aligned(
                            src.offset(offset as isize),
                            dst.offset(offset as isize),
                        );
                        offset += SIMD_STRIDE_SIZE;
                        if offset > len_minus_stride_times_two {
                            break;
                        }
                    }
                }
            }
            if offset < len_minus_stride {
                unpack_stride_src_aligned(src.offset(offset as isize), dst.offset(offset as isize));
                offset += SIMD_STRIDE_SIZE;
            }
        }
        while offset < len {
            let code_unit = *(src.offset(offset as isize));
            // On x86_64, this loop autovectorizes but in the pack
            // case there are instructions whose purpose is to make sure
            // each u16 in the vector is truncated before packing. However,
            // since we don't care about saturating behavior of SSE2 packing
            // when the input isn't Latin1, those instructions are useless.
            // Unfortunately, using the `assume` intrinsic to lie to the
            // optimizer doesn't make LLVM omit the trunctation that we
            // don't need. Possibly this loop could be manually optimized
            // to do the sort of thing that LLVM does but without the
            // ANDing the read vectors of u16 with a constant that discards
            // the high half of each u16. As far as I can tell, the
            // optimization assumes that doing a SIMD read past the end of
            // the array is OK.
            *(dst.offset(offset as isize)) = code_unit as u16;
            offset += 1;
        }
    }

    #[inline(never)]
    pub fn convert_latin1_to_utf16(src: &[u8], dst: &mut [u16]) {
        assert!(
            dst.len() >= src.len(),
            "Destination must not be shorter than the source."
        );
        // TODO: On aarch64, the safe version autovectorizes to the same unpacking
        // instructions and this code, but, yet, the autovectorized version is
        // faster.
        unsafe {
            unpack_latin1(src.as_ptr(), dst.as_mut_ptr(), src.len());
        }
    }

    #[bench]
    fn bench_mem_convert_latin1_to_utf16_1000(b: &mut Bencher) {
        let bytes = include_bytes!("wikipedia/de-edit.txt");
        let mut v = Vec::with_capacity(bytes.len());
        v.extend_from_slice(bytes);
        let truncated = &v[..1000];
        let capacity = 1000 * 4;
        let mut vec = Vec::with_capacity(capacity);
        vec.resize(capacity, 0u16);
        let dst = &mut vec[..];
        b.bytes = (truncated.len() as u64) * MEM_INNER_LOOP;
        b.iter(|| {
            for _ in 0..MEM_INNER_LOOP {
                test::black_box(convert_latin1_to_utf16(test::black_box(truncated), test::black_box(dst)));
            }
        });
    }

}

