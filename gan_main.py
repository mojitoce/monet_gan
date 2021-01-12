import matplotlib.pyplot as plt
import gan_functions as gf
import keras
import tensorflow as tf



monet_im = gf.import_data('monet_jpg')
photo_im = gf.import_data('photo_jpg')


monet_train, monet_test = gf.train_test(monet_im, 0.8)
photo_train, photo_test = gf.train_test(photo_im, 0.8)



# Generator
OUTPUT_CHANNELS = 3



monet_generator = gf.Generator() # transforms photos to Monet-esque paintings
photo_generator = gf.Generator() # transforms Monet paintings to be more like photos

monet_discriminator = gf.Discriminator() # differentiates real Monet paintings and generated Monet paintings
photo_discriminator = gf.Discriminator() # differentiates real photos and generated photos



class CycleGan(keras.Model):
    def __init__(self, monet_g, photo_g, monet_d, photo_d, lambda_cycle):

        super(CycleGan, self).__init__()
        self.monet_g = monet_g
        self.photo_g = photo_g
        self.monet_d = monet_d
        self.photo_d = photo_d
        self.lambda_cycle = lambda_cycle

    def compile(self, m_gen_optimizer, p_gen_optimizer, m_disc_optimizer,
        p_disc_optimizer, gen_loss_fn, disc_loss_fn, cycle_loss_fn, identity_loss_fn):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        print(real_monet, type(real_monet))
        with tf.GradientTape(persistent=True) as tape:
            print('Break 0')
            # Real monet to fake photo
            fake_photo = self.photo_g(real_monet, training = True)
            print('Break 0.1')
            # Real photo to fake Monet
            fake_monet = self.monet_g(real_photo, training = True)
            print('Break 0.2')
            # Full cycle
            tt_monet = self.monet_g(fake_photo, training = True)
            tt_photo = self.photo_g(fake_monet, training = True)

            # Preserves identity
            id_monet = self.monet_g(real_monet, training = True)
            id_photo = self.photo_g(real_photo, training = True)
            print('Break 1')

            # Discriminator
            disc_real_monet = self.monet_d(real_monet, training = True)
            disc_fake_monet = self.monet_d(fake_monet, training = True)

            disc_real_photo = self.photo_d(real_photo, training = True)
            disc_fake_photo = self.photo_d(fake_photo, training = True)
            print('Break 2')
            print(disc_fake_monet, disc_real_monet, tf.ones_like(disc_fake_monet))

            # Evaluate generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            print('Break 2.1')
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)
            print('Break 3')
            # Evaluate cycle loss
            monet_cycle_loss = self.cycle_loss_fn(real_monet, tt_monet, self.lambda_cycle)
            photo_cycle_loss = self.cycle_loss_fn(real_photo, tt_photo, self.lambda_cycle)
            total_cycle_loss = monet_cycle_loss + photo_cycle_loss
            print('Break 4')
            # Evaluate identity loss
            monet_id_loss = self.identity_loss_fn(real_monet, id_monet, self.lambda_cycle)
            photo_id_loss = self.identity_loss_fn(real_photo, id_photo, self.lambda_cycle)

            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + monet_id_loss
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + photo_id_loss

            # Evaluate discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)


        monet_gen_grad = tape.gradient(total_monet_gen_loss, self.monet_g.trainable_variables)
        photo_gen_grad = tape.gradient(total_photo_gen_loss, self.photo_g.trainable_variables)

        monet_disc_grad = tape.gradient(monet_disc_loss, self.monet_d.trainable_variables)
        photo_disc_grad = tape.gradient(photo_disc_loss, self.photo_d.trainable_variables)

        self.m_gen_optimizer.apply_gradients(zip(monet_gen_grad, self.monet_g.trainable_variables))
        self.p_gen_optimizer.apply_gradients(zip(photo_gen_grad, self.photo_g.trainable_variables))
        self.m_disc_optimizer.apply_gradients(zip(monet_disc_grad, self.monet_d.trainable_variables))
        self.p_disc_optimizer.apply_gradients(zip(photo_disc_grad, self.photo_d.trainable_variables))

        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }


# Optimizers
m_gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
p_gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

m_disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
p_disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# Compile model
cycle_gan_model = CycleGan(
        monet_generator, photo_generator, monet_discriminator, photo_discriminator, lambda_cycle=10.0
    )

cycle_gan_model.compile(
    m_gen_optimizer = m_gen_optimizer,
    p_gen_optimizer = p_gen_optimizer,
    m_disc_optimizer = m_disc_optimizer,
    p_disc_optimizer = p_disc_optimizer,
    gen_loss_fn = gf.generator_loss,
    disc_loss_fn = gf.discriminator_loss,
    cycle_loss_fn = gf.cycle_loss,
    identity_loss_fn = gf.id_loss
)


cycle_gan_model.fit(
    tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(monet_im).batch(1),
                         tf.data.Dataset.from_tensor_slices(photo_im).batch(1))),
    epochs=10
)
