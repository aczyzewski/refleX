function updateFooter() {
    var footerH = $("#cmb-footer").height();
    var contentH = $("#cmb-header").height() + $("#cmb-content").height() + 40 + 10 + 20;
    var windowH = $(window).height();
    if (contentH + footerH > windowH) {
        $("#cmb-footer").removeClass("fixed");
    } else {
        $("#cmb-footer").addClass("fixed");
    }
}

$(document)
    .on('change', '.btn-file :file', function () {
        var input = $(this),
            label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
        console.log(label);
        input.parent().find('.filename-holder').html(label);
    });

$(document).ready(function () {
    $(window).resize(function () {
        updateFooter();
    });

    $('#id_application').change(function() {
        setTimeout(function(){ updateFooter(); },30)
    });

    updateFooter();
});
