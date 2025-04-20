from PIL import Image
import arcade


def gif_to_texturegrid(file):
    try:
        gif = Image.open(file)
        inSsheet = Image.new("RGBA", (gif.size[0] * gif.n_frames, gif.size[1]))
        for frame in range(0,gif.n_frames):
            gif.seek(frame)
            extracted_frame = gif.resize(gif.size)
            position = (gif.size[0]*frame, 0)
            inSsheet.paste(extracted_frame, position)
        Ssheet = arcade.SpriteSheet.from_image(inSsheet)
        output = Ssheet.get_texture_grid(size=gif.size, columns=gif.n_frames, count=gif.n_frames)
        return output
    except Exception as e:
        print(e)


class gifSprite(arcade.Sprite):
    def __init__(self, gif_path=None, onetime=False, speed=1, texture_list=None):
        """
        :param gif_path: path to gif
        :param onetime: should the gif play one time?
        :param speed: gif speed, bigger = faster, default 1
        """
        if texture_list is None:
            texture_list = gif_to_texturegrid(gif_path)
        super().__init__(texture_list[0])
        self.time_elapsed = 0
        self.current_texture = 0
        self.textures = texture_list
        self.speed = speed
        self.onetime = onetime

    def update(self, delta_time=1 / 60):
        self.time_elapsed += delta_time
        if self.current_texture < len(self.textures):
            self.set_texture(self.current_texture)
        if self.current_texture == len(self.textures):
            if self.onetime:
                self.remove_from_sprite_lists()
            self.time_elapsed = 0
        self.current_texture = int(self.time_elapsed * (60*self.speed))

# class GameView(arcade.View):  # how to use
#     def __init__(self):
#         super().__init__()
#         self.explosions_list = arcade.SpriteList()
#         self.background_color = arcade.color.BLACK
#
#     def on_draw(self):
#         self.clear()
#         self.explosions_list.draw()
#
#     def on_key_press(self, symbol: int, modifiers: int):
#         if symbol == arcade.key.F:
#             explosion = gifSprite('explosion.gif', True)
#             explosion.center_x = self.window.center_x
#             explosion.center_y = self.window.center_y
#             explosion.update()
#             self.explosions_list.append(explosion)
#
#     def on_update(self, delta_time):
#         self.explosions_list.update(delta_time)
#
#
# def main():
#     window = arcade.Window(1000, 500, 'test')
#     game = GameView()
#     window.show_view(game)
#     arcade.run()
#
# if __name__ == "__main__":
#     main()