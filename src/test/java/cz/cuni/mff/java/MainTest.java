package cz.cuni.mff.java;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class MainTest {

    @Test
    public void testDefaultArgs() {
        String[] args = {};
        Map<String, String> flags = invokeParseArgs(args);
        assertEquals("true", flags.getOrDefault("evaluate", "true"));
        assertEquals("true", flags.getOrDefault("plot", "true"));
        assertEquals("true", flags.getOrDefault("save", "true"));
    }

    @Test
    public void testAllFlagsFalse() {
        String[] args = {
                "--evaluate=false",
                "--plot=false",
                "--save=false"
        };
        Map<String, String> flags = invokeParseArgs(args);
        assertEquals("false", flags.get("evaluate"));
        assertEquals("false", flags.get("plot"));
        assertEquals("false", flags.get("save"));
    }

    @Test
    public void testCustomModelAndDataPaths() {
        String[] args = {
                "--modelPath=out/custom_model.ser",
                "--dataPath=data/custom.csv",
                "--splitRatio=0.9"
        };
        Map<String, String> flags = invokeParseArgs(args);
        assertEquals("out/custom_model.ser", flags.get("modelPath"));
        assertEquals("data/custom.csv", flags.get("dataPath"));
        assertEquals("0.9", flags.get("splitRatio"));
    }

    // Helper to access private method
    private Map<String, String> invokeParseArgs(String[] args) {
        try {
            var method = Main.class.getDeclaredMethod("parseArgs", String[].class);
            method.setAccessible(true);
            return (Map<String, String>) method.invoke(null, (Object) args);
        } catch (Exception e) {
            throw new RuntimeException("Failed to invoke parseArgs", e);
        }
    }
}
