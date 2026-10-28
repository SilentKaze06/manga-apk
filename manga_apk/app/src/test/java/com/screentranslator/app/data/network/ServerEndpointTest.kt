package com.screentranslator.app.data.network

import org.junit.Assert.*
import org.junit.Test

class ServerEndpointTest {
    @Test fun omittedSchemeUsesHttpsAndCanonicalOrigin() {
        assertEquals("https://example.com/api", ServerEndpoint.normalize(" EXAMPLE.com:443/api/ "))
        assertTrue(ServerEndpoint.matches("https://example.com/", "example.com:443"))
        assertFalse(ServerEndpoint.matches("https://old.example", "https://new.example"))
        assertFalse(ServerEndpoint.matches("https://example.com/a", "https://example.com/b"))
    }
    @Test fun releaseRejectsAllCleartext() {
        for (host in listOf("10.0.2.2", "192.168.1.20", "localhost", "example.com")) {
            assertThrows(IllegalArgumentException::class.java) { ServerEndpoint.normalize("http://$host:8000") }
        }
    }
    @Test fun debugOnlyAllowsExplicitLocalCleartext() {
        assertEquals("http://10.0.2.2:8000", ServerEndpoint.normalize("http://10.0.2.2:8000/", true))
        assertEquals("https://10.0.2.2:8000", ServerEndpoint.normalize("10.0.2.2:8000", true))
        for (host in listOf("example.com", "8.8.8.8", "172.32.0.1", "192.169.0.1")) {
            assertThrows(IllegalArgumentException::class.java) { ServerEndpoint.normalize("http://$host", true) }
        }
    }
    @Test fun rejectsAmbiguousOrCredentialBearingEndpoints() {
        for (url in listOf("", "ftp://example.com", "https://user:password@example.com", "https://example.com/?token=x", "https://example.com/#x")) {
            assertThrows(IllegalArgumentException::class.java) { ServerEndpoint.normalize(url) }
        }
    }
}
