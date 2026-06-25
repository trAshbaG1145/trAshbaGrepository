package com.project.yuhangvue;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.UUID;

@SpringBootTest
class YuhangVueApplicationTests {

    @Test
    public void testPage() {
        for (int i = 0; i < 10; i++) {
            System.out.println(UUID.randomUUID().toString());
        }

    }

}
