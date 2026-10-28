package com.screentranslator.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import com.screentranslator.app.ui.settings.SettingsScreen
import com.screentranslator.app.ui.settings.SettingsViewModel
import com.screentranslator.app.ui.theme.ScreenTranslatorTheme

class MainActivity : ComponentActivity() {

    private val viewModel: SettingsViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            ScreenTranslatorTheme {
                SettingsScreen(
                    viewModel = viewModel
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        viewModel.refreshHistory()
    }
}
